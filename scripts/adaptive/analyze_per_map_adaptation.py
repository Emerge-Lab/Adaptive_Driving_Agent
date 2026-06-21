"""Per-map adaptation analysis across the gb=3 legacy-eval-fix k×seed sweep.

For every run tagged ada_k{K}_gb3_legacy_eval_fix:
  - pull the LATEST eval_maps/per_map_summary table (one row per map:
    map_id, t0, t1[, t2, ...], ada_delta_last_minus_0)
  - compute: eval score, mean ada_delta, # maps positive / negative / neutral
  - dump per-map CSV so individual maps can be inspected (+ video later)

Outputs (under --out):
  per_run_summary.csv        one row per (k, seed): means + distribution counts
  per_map_<k>_seed<s>.csv     full per-map table for each run
  ada_delta_dist_by_k.png     histogram of per-map ada_delta, faceted by k
  ada_delta_vs_k.png          mean ada_delta vs k with per-seed points
  top_adapting_maps.csv       maps with strongest +delta (for video inspection)
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


THRESH = 0.05  # |delta| > THRESH counts as adapted / regressed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="emerge_")
    ap.add_argument("--project", default="adaptive_aligned_v2")
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    ap.add_argument("--out", type=Path, default=Path("outputs/per_map_adaptation"))
    return ap.parse_args()


def fetch_per_map_table(run):
    """Return list of rows + columns from the latest per_map_summary artifact."""
    best = None
    for art in run.logged_artifacts():
        if "per_map_summary" in art.name:
            # version number after ':v'
            ver = int(art.version.lstrip("v")) if art.version.lstrip("v").isdigit() else 0
            if best is None or ver > best[0]:
                best = (ver, art)
    if best is None:
        return None, None
    art = best[1]
    d = art.download(root=f"/tmp/wb_permap/{run.id}")
    tj = next(Path(d).rglob("*.json"))
    with open(tj) as f:
        data = json.load(f)
    return data["columns"], data["data"]


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    per_run = []          # (k, seed, eval_score, mean_delta, n_pos, n_neg, n_neu, n_maps)
    per_map_by_k = defaultdict(list)   # k -> list of all per-map deltas (pooled across seeds)
    top_maps = []         # (k, seed, map_id, t0, t_last, delta)

    for k in args.ks:
        tag = f"ada_k{k}_gb3_legacy_eval_fix"
        runs = api.runs(f"{args.entity}/{args.project}", filters={"tags": tag})
        print(f"\n=== {tag}: {len(runs)} runs ===")
        for run in runs:
            seed = run.config.get("train", {}).get("seed", run.config.get("seed", "?"))
            cols, rows = fetch_per_map_table(run)
            if cols is None:
                print(f"  {run.id} seed={seed}: NO per_map table")
                continue
            # delta column is the last one (ada_delta_last_minus_0)
            di = cols.index("ada_delta_last_minus_0")
            t0i = cols.index("t0")
            tlast_name = [c for c in cols if c.startswith("t") and c[1:].isdigit()][-1]
            tli = cols.index(tlast_name)
            mi = cols.index("map_id")
            deltas = np.array([r[di] for r in rows], dtype=float)
            t0 = np.array([r[t0i] for r in rows], dtype=float)
            tl = np.array([r[tli] for r in rows], dtype=float)
            eval_score = float(np.mean([t0.mean(), tl.mean()]))
            n_pos = int((deltas > THRESH).sum())
            n_neg = int((deltas < -THRESH).sum())
            n_neu = len(deltas) - n_pos - n_neg
            per_run.append((k, seed, run.id, eval_score, float(deltas.mean()),
                            n_pos, n_neg, n_neu, len(deltas)))
            per_map_by_k[k].extend(deltas.tolist())
            print(f"  {run.id} seed={seed}: n_maps={len(deltas)} "
                  f"mean_delta={deltas.mean():+.4f}  +{n_pos} / -{n_neg} / ={n_neu}")

            # full per-map dump
            with open(args.out / f"per_map_k{k}_seed{seed}_{run.id}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)

            # top adapting maps (delta > THRESH), sorted desc
            order = np.argsort(-deltas)
            for idx in order:
                if deltas[idx] <= THRESH:
                    break
                top_maps.append((k, seed, int(rows[idx][mi]),
                                 float(t0[idx]), float(tl[idx]), float(deltas[idx])))

    # per_run_summary.csv
    with open(args.out / "per_run_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "seed", "run_id", "eval_score", "mean_ada_delta",
                    "n_pos", "n_neg", "n_neutral", "n_maps"])
        w.writerows(per_run)

    # top_adapting_maps.csv (sorted by delta desc within k)
    top_maps.sort(key=lambda x: (x[0], -x[5]))
    with open(args.out / "top_adapting_maps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "seed", "map_id", "t0", "t_last", "ada_delta"])
        w.writerows(top_maps)

    # Plot 1: ada_delta distribution faceted by k
    ks = sorted(per_map_by_k.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4), squeeze=False)
    for i, k in enumerate(ks):
        d = np.array(per_map_by_k[k])
        ax = axes[0, i]
        ax.hist(d, bins=40, range=(-1, 1), color="tab:green", edgecolor="black")
        ax.axvline(0, color="black", linestyle=":")
        ax.axvline(d.mean(), color="red", linestyle="--", label=f"mean {d.mean():+.3f}")
        npos = int((d > THRESH).sum()); nneg = int((d < -THRESH).sum())
        ax.set_title(f"k={k}  (pooled {len(d)} map-evals)\n+{npos} / -{nneg}")
        ax.set_xlabel("per-map ada_delta (t_last - t0)")
        ax.legend(fontsize=8)
    fig.suptitle("Per-map adaptation distribution by k (gb=3 legacy-eval-fix)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out / "ada_delta_dist_by_k.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: mean ada_delta vs k (per-seed points + mean)
    fig, ax = plt.subplots(figsize=(8, 5))
    by_k = defaultdict(list)
    for (k, seed, rid, es, md, *_rest) in per_run:
        by_k[k].append(md)
        ax.scatter(k, md, color="tab:blue", alpha=0.6, s=40)
    xs = sorted(by_k.keys())
    means = [np.mean(by_k[k]) for k in xs]
    ax.plot(xs, means, "r-o", label="mean across seeds")
    ax.axhline(0, color="black", linestyle=":")
    ax.set_xlabel("k (number of trials)")
    ax.set_ylabel("mean eval ada_delta (t_last - t0)")
    ax.set_title("Eval ada_delta vs k — does adaptation grow with trial count?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "ada_delta_vs_k.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote analysis to {args.out}")
    print(f"  per_run_summary.csv, top_adapting_maps.csv ({len(top_maps)} rows)")
    print(f"  ada_delta_dist_by_k.png, ada_delta_vs_k.png")


if __name__ == "__main__":
    main()
