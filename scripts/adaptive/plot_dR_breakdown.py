"""ΔR breakdown into GOAL / COLLISION / OFFROAD / LANE.

All four components sourced directly from the env-side per-trial accumulators
(drive.h trial_R_*), snapshotted in the evaluator at each trial_ended_this_step.
Replaces the earlier post-hoc threshold decomp (which merged collision+offroad
into "crash" and dropped lane because the threshold logic was wrong).

Sum-check: per-cell mean(ΔR_goal + ΔR_collision + ΔR_offroad + ΔR_lane) should
equal mean(ΔR_total) within float-precision (modulo the rare goal-overwrites-crash
edge case which contributes ≤ 0.25 to the per-cell mean — flagged below).

Filtered to adaptable maps (p_0 < 0.8). Per cell shows:
- absolute ΔR contribution of each component
- % of ΔR_total carried by each component

Outputs:
  outputs/eval540_return/g_dR_breakdown_room.jpg  -- 4-panel heatmap
"""
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_DIR = Path("outputs/eval540_return")
SUCCESS_CSV = Path("outputs/eval540_combined/all_cells.csv")
MANIFEST = Path("scripts/adaptive/final_runs_manifest.csv")
OUT_DIR = Path("outputs/eval540_return")
SOLO_ENTROPY = 0.10

ENTROPIES = [0.05, 0.10, 0.20, 0.50]
KS = [2, 3, 4, 5, 6]
SEEDS = [42, 43, 44]


def _load_metric(prefix):
    """data[ent][k][seed][mid] = tuple of length k."""
    pat = re.compile(rf"per_map_{re.escape(prefix)}_k(\d+)_seed(\d+)_([0-9a-z]+)\.csv$")
    manifest = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            manifest[r["wandb_id"]] = float(r["entropy_ub"])
    data = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(str(IN_DIR / f"per_map_{prefix}_k*_seed*.csv"))):
        m = pat.search(Path(path).name)
        if m is None:
            continue
        k, seed, wid = int(m.group(1)), int(m.group(2)), m.group(3)
        ent = manifest.get(wid, SOLO_ENTROPY)
        with open(path) as fh:
            for r in csv.DictReader(fh):
                mid = int(float(r["map_id"]))
                ts = tuple(float(r[f"t{i}"]) for i in range(k))
                data[ent][k].setdefault(seed, {})[mid] = ts
    return data


def load_p0_keep(threshold=0.8):
    p0 = defaultdict(lambda: defaultdict(dict))
    with open(SUCCESS_CSV) as f:
        for r in csv.DictReader(f):
            ent = float(r["entropy_ub"]); k = int(r["k"]); seed = int(r["seed"])
            p0[(ent, k)][seed][int(r["map_id"])] = float(r["t0"])
    keep = {}
    for cell, by_seed in p0.items():
        seeds = list(by_seed)
        common = sorted(set.intersection(*(set(by_seed[s]) for s in seeds)))
        mean_p0 = {m: float(np.mean([by_seed[s][m] for s in seeds])) for m in common}
        keep[cell] = {m for m, p in mean_p0.items() if p < threshold}
    return keep


def _delta_per_seed(data, ent, k, allowed):
    """Returns sorted-mids -> (seed -> delta_array) restricted to allowed maps."""
    out = {}
    for seed in SEEDS:
        rows = data[ent][k].get(seed, {})
        if not rows:
            continue
        sel = sorted(m for m in rows if m in allowed)
        if not sel:
            continue
        arr = np.array([rows[m] for m in sel])
        out[seed] = (sel, arr[:, -1] - arr[:, 0])
    return out


def _common_stack(per_seed_delta):
    """Stack (S, M_common) across seeds intersected on mids."""
    if len(per_seed_delta) != len(SEEDS):
        return None
    seeds_sorted = sorted(per_seed_delta)
    sets = [set(per_seed_delta[s][0]) for s in seeds_sorted]
    common = sorted(set.intersection(*sets))
    if not common:
        return None
    stack = np.stack([
        np.array([per_seed_delta[s][1][per_seed_delta[s][0].index(m)] for m in common])
        for s in seeds_sorted
    ])
    return stack, common


def cell_means(data_R, data_goal, data_collision, data_offroad, data_lane, keep):
    """Per cell: mean ΔR_total + 4 components. All on the SAME common adaptable
    map set (intersected across all 5 metrics × all 3 seeds)."""
    out = {}
    metrics = [("R", data_R), ("goal", data_goal), ("collision", data_collision),
               ("offroad", data_offroad), ("lane", data_lane)]
    for ent in ENTROPIES:
        for k in KS:
            allowed = keep.get((ent, k), set())
            per_seed_by_metric = {
                name: _delta_per_seed(d, ent, k, allowed) for name, d in metrics
            }
            if not all(per_seed_by_metric[name] for name, _ in metrics):
                continue
            # Intersect map_ids across all metrics × all seeds.
            all_sets = []
            ok = True
            for name, _ in metrics:
                for s in SEEDS:
                    if s not in per_seed_by_metric[name]:
                        ok = False; break
                    all_sets.append(set(per_seed_by_metric[name][s][0]))
                if not ok:
                    break
            if not ok:
                continue
            common = sorted(set.intersection(*all_sets))
            if not common:
                continue
            def _stack(ps):
                return np.stack([
                    np.array([ps[s][1][ps[s][0].index(m)] for m in common])
                    for s in sorted(ps)
                ])
            means = {
                name: float(_stack(per_seed_by_metric[name]).mean(axis=0).mean())
                for name, _ in metrics
            }
            out[(ent, k)] = dict(**means, n=len(common))
    return out


def plot_breakdown(cells, fname):
    panels = [
        ("goal",      "ΔR_goal (more goal-reaches)"),
        ("collision", "ΔR_collision (fewer vehicle collisions)"),
        ("offroad",   "ΔR_offroad (less time offroad)"),
        ("lane",      "ΔR_lane (per-step lane shaping)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))
    # shared symmetric color scale per panel
    for ax_i, (key, title) in enumerate(panels):
        M = np.full((len(ENTROPIES), len(KS)), np.nan)
        Pct = np.full_like(M, np.nan)
        Ns = np.zeros_like(M, dtype=int)
        for i, ent in enumerate(ENTROPIES):
            for j, k in enumerate(KS):
                c = cells.get((ent, k))
                if c is None:
                    continue
                M[i, j] = c[key]
                Ns[i, j] = c["n"]
                if abs(c["R"]) > 1e-6:
                    Pct[i, j] = 100.0 * c[key] / c["R"]
        ax = axes[ax_i]
        vmax = max(0.05, float(np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
        ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
        ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
        for i in range(len(ENTROPIES)):
            for j in range(len(KS)):
                if np.isnan(M[i, j]):
                    continue
                pct = Pct[i, j]
                txt = f"{M[i,j]:+.02f}\n({pct:+.0f}%)"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label=f"ΔR_{key}")
        ax.set_title(title, fontsize=11)
    fig.suptitle("ΔR breakdown on adaptable maps (p_0 < 0.8). "
                 "Numbers: absolute Δ and (% of ΔR_total). "
                 "Components from env-side per-trial accumulators (sum ≈ R).",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def print_table(cells):
    print("\n=== ΔR breakdown on adaptable maps (p_0 < 0.8) ===")
    print(f"{'ent':>5} {'k':>2} {'n':>4} {'ΔR':>8} | "
          f"{'goal':>7} {'%g':>4} | {'coll':>8} {'%c':>4} | "
          f"{'off':>8} {'%o':>4} | {'lane':>7} {'%l':>4} | "
          f"{'sum':>8} {'diff':>7}")
    for ent in ENTROPIES:
        for k in KS:
            c = cells.get((ent, k))
            if c is None:
                continue
            R = c["R"]
            def _pct(x):
                return (100.0 * x / R) if abs(R) > 1e-6 else float("nan")
            s = c["goal"] + c["collision"] + c["offroad"] + c["lane"]
            print(f"{ent:>5.2f} {k:>2} {c['n']:>4} {R:>+8.3f} | "
                  f"{c['goal']:>+7.3f} {_pct(c['goal']):>+4.0f} | "
                  f"{c['collision']:>+8.3f} {_pct(c['collision']):>+4.0f} | "
                  f"{c['offroad']:>+8.3f} {_pct(c['offroad']):>+4.0f} | "
                  f"{c['lane']:>+7.3f} {_pct(c['lane']):>+4.0f} | "
                  f"{s:>+8.3f} {R - s:>+7.3f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_R         = _load_metric("R")
    data_goal      = _load_metric("goal")
    data_collision = _load_metric("collision")
    data_offroad   = _load_metric("offroad")
    data_lane      = _load_metric("lane")
    keep = load_p0_keep(threshold=0.8)
    cells = cell_means(data_R, data_goal, data_collision, data_offroad, data_lane, keep)
    plot_breakdown(cells, "g_dR_breakdown_room.jpg")
    print_table(cells)


if __name__ == "__main__":
    main()
