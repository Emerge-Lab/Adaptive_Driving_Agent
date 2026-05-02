"""Score each nuplan map for vehicle-vehicle interaction density.

For each map, count pairs of valid moving vehicles that come within
PROXIMITY_M of each other at the same timestep. Sum over all (pair, timestep)
gives a "hardness score" — maps with high score have lots of vehicle
interaction; maps with low score are mostly drive-straight.

Output: CSV with [bin_id, scenario_id, num_vehicles, num_valid_vehicles,
                  total_steps, interaction_events, score_per_step,
                  unique_pairs_in_interaction]

Note on bin_id: JSONs are sorted alphabetically and mapped to map_{i:03d}.bin
where i starts at 1 (verified empirically from existing files in
resources/drive/binaries/nuplan_201/). Sort order in this script must match.
"""
import os, sys, json, glob, argparse, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

PROXIMITY_M = 5.0   # meters
V_MIN = 0.5         # m/s — exclude parked vehicles
INVALID_SENTINEL = -10000.0


def score_map(json_path, vehicle_only=True):
    with open(json_path) as f:
        d = json.load(f)
    objs_all = d.get("objects", [])
    if vehicle_only:
        objs = [o for o in objs_all if o.get("type") == "vehicle"]
    else:
        objs = objs_all

    if len(objs) < 2:
        return {
            "scenario_id": d.get("scenario_id", "?"),
            "num_vehicles": len(objs_all),
            "num_valid_vehicles": len(objs),
            "total_steps": 0,
            "interaction_events": 0,
            "sdc_interaction_steps": 0,
            "sdc_interactions_total": 0,
            "unique_pairs_in_interaction": 0,
            "score_per_step": 0.0,
        }

    T = len(objs[0]["position"])
    N = len(objs)
    sdc_idx = None
    for i, o in enumerate(objs):
        if o.get("is_sdc"):
            sdc_idx = i
            break

    # Build (T, N) arrays of x, y, vx, vy, valid
    px = np.full((T, N), INVALID_SENTINEL, dtype=np.float32)
    py = np.full((T, N), INVALID_SENTINEL, dtype=np.float32)
    vx = np.full((T, N), 0.0, dtype=np.float32)
    vy = np.full((T, N), 0.0, dtype=np.float32)
    valid = np.zeros((T, N), dtype=bool)
    for i, o in enumerate(objs):
        for t in range(T):
            if o["valid"][t]:
                pos = o["position"][t]
                vel = o["velocity"][t]
                px[t, i] = pos["x"]
                py[t, i] = pos["y"]
                vx[t, i] = vel.get("x", 0.0)
                vy[t, i] = vel.get("y", 0.0)
                valid[t, i] = True

    speed = np.sqrt(vx ** 2 + vy ** 2)
    moving = speed > V_MIN

    interaction_events = 0
    sdc_interaction_steps = 0
    sdc_interactions_total = 0
    pairs_seen = set()

    for t in range(T):
        idx = np.where(valid[t] & moving[t])[0]
        if len(idx) < 2:
            continue
        x = px[t, idx]
        y = py[t, idx]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        iu, ju = np.triu_indices(len(idx), k=1)
        close = dist[iu, ju] < PROXIMITY_M
        n_close = int(close.sum())
        interaction_events += n_close
        if n_close > 0:
            close_pair_idx = np.where(close)[0]
            for k in close_pair_idx:
                a, b = idx[iu[k]], idx[ju[k]]
                pairs_seen.add((min(a, b), max(a, b)))

        # SDC-specific: count this step if SDC is present, moving, and has any
        # vehicle within proximity at this step
        if sdc_idx is not None and sdc_idx in idx:
            sdc_pos_t = (px[t, sdc_idx], py[t, sdc_idx])
            others = [j for j in idx if j != sdc_idx]
            if others:
                ox = px[t, others]
                oy = py[t, others]
                d_sdc = np.sqrt((ox - sdc_pos_t[0])**2 + (oy - sdc_pos_t[1])**2)
                n_close_sdc = int((d_sdc < PROXIMITY_M).sum())
                if n_close_sdc > 0:
                    sdc_interaction_steps += 1
                    sdc_interactions_total += n_close_sdc

    return {
        "scenario_id": d.get("scenario_id", "?"),
        "num_vehicles": len(objs_all),
        "num_valid_vehicles": N,
        "total_steps": T,
        "interaction_events": int(interaction_events),
        "sdc_interaction_steps": int(sdc_interaction_steps),
        "sdc_interactions_total": int(sdc_interactions_total),
        "unique_pairs_in_interaction": len(pairs_seen),
        "score_per_step": float(interaction_events) / max(T, 1),
    }


def process_one(args):
    bin_id, path = args
    try:
        result = score_map(path)
        result["bin_id"] = bin_id
        result["json_path"] = path
        return result
    except Exception as e:
        return {"bin_id": bin_id, "json_path": path, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/ADA/data/nuplan_gpudrive/nuplan")
    ap.add_argument("--out", default="/tmp/nuplan_201_hardness_scores.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all maps")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if args.limit > 0:
        json_files = json_files[:args.limit]
    print(f"Found {len(json_files)} maps. Workers={args.workers}")

    # bin_id starts at 1 (matching existing map_001.bin onwards)
    tasks = [(i + 1, p) for i, p in enumerate(json_files)]

    rows = []
    n_done = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(process_one, tasks, chunksize=20):
            n_done += 1
            if "error" in r:
                n_err += 1
                if n_err < 5:
                    print(f"  err on {r['json_path']}: {r['error']}")
                continue
            rows.append(r)
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(tasks)} done")

    # Save CSV
    cols = ["bin_id", "scenario_id", "num_vehicles", "num_valid_vehicles",
            "total_steps", "interaction_events", "sdc_interaction_steps",
            "sdc_interactions_total", "unique_pairs_in_interaction",
            "score_per_step", "json_path"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n{len(rows)} maps scored, {n_err} errors. Saved → {args.out}")

    # Distribution summary for both metrics
    for metric_key, metric_label in [
        ("interaction_events", "All vehicle pairs"),
        ("sdc_interaction_steps", "SDC interaction steps (steps where SDC has any other moving vehicle nearby)"),
    ]:
        scores = np.array([r[metric_key] for r in rows])
        print(f"\n=== {metric_label} ===")
        print(f"  min:  {scores.min()}")
        print(f"  25%:  {int(np.percentile(scores, 25))}")
        print(f"  50%:  {int(np.percentile(scores, 50))}")
        print(f"  75%:  {int(np.percentile(scores, 75))}")
        print(f"  90%:  {int(np.percentile(scores, 90))}")
        print(f"  95%:  {int(np.percentile(scores, 95))}")
        print(f"  max:  {scores.max()}")
        print(f"  mean: {scores.mean():.1f}")
        print(f"  zeros: {(scores == 0).sum()} maps")


if __name__ == "__main__":
    main()
