"""Two clean adaptation plots from the cached per-map CSVs.

1. ada_delta vs k: mean +/- std band across seeds + individual run points.
2. per-trial score faceted by k: for each k, trial-0..k-1 success rate,
   mean +/- std across seeds.

Reads outputs/per_map_adaptation/per_map_k{k}_seed{s}_{rid}.csv
(columns: map_id, t0, t1, ..., t_{k-1}, ada_delta_last_minus_0)
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("outputs/per_map_adaptation")
OUT = SRC
FNAME_RE = re.compile(r"per_map_k(\d+)_seed(\d+)_(\w+)\.csv")
# Degraded runs to exclude (never converged): 33od5iqw = k4 seed43 that ran at
# ~5K SPS from GPU contention, only reached 474M steps / score 0.33. A clean
# rerun (ufmegw4l) of the same config exists.
EXCLUDE = {"33od5iqw"}


def load_runs():
    """Return {k: [ {seed, run_id, deltas(np), trial_scores(dict trial->mean)} ]}."""
    runs = defaultdict(list)
    for f in sorted(SRC.glob("per_map_k*_seed*.csv")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        k, seed, rid = int(m.group(1)), int(m.group(2)), m.group(3)
        if rid in EXCLUDE:
            continue
        with open(f) as fh:
            r = csv.DictReader(fh)
            cols = r.fieldnames
            rows = list(r)
        trial_cols = [c for c in cols if c.startswith("t") and c[1:].isdigit()]
        trial_cols.sort(key=lambda c: int(c[1:]))
        deltas = np.array([float(x["ada_delta_last_minus_0"]) for x in rows])
        trial_means = {int(c[1:]): float(np.mean([float(x[c]) for x in rows])) for c in trial_cols}
        runs[k].append({"seed": seed, "run_id": rid, "deltas": deltas, "trials": trial_means})
    return runs


def plot_ada_delta_vs_k(runs):
    ks = sorted(runs.keys())
    means, stds = [], []
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k in ks:
        per_run_mean = [np.mean(r["deltas"]) for r in runs[k]]
        m, s = np.mean(per_run_mean), np.std(per_run_mean)
        means.append(m); stds.append(s)
        # individual run points
        ax.scatter([k] * len(per_run_mean), per_run_mean, color="0.5", s=28, zorder=3,
                   label="per-seed" if k == ks[0] else None)
    means = np.array(means); stds = np.array(stds)
    ax.plot(ks, means, color="black", lw=1.5, zorder=2, label="mean")
    ax.fill_between(ks, means - stds, means + stds, color="black", alpha=0.15, zorder=1, label="±1 std")
    ax.axhline(0, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("k")
    ax.set_ylabel("eval ada_delta (t_last − t0)")
    ax.set_xticks(ks)
    ax.set_title("ada_delta vs k")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "ada_delta_vs_k_clean.jpg", dpi=140, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    print("saved ada_delta_vs_k_clean.jpg")


def plot_per_trial_by_k(runs):
    ks = sorted(runs.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4), squeeze=False, sharey=True)
    for i, k in enumerate(ks):
        ax = axes[0, i]
        trials = sorted(runs[k][0]["trials"].keys())
        # gather per-trial scores across seeds
        mat = np.array([[r["trials"][t] for t in trials] for r in runs[k]])  # (n_seed, n_trial)
        m = mat.mean(axis=0); s = mat.std(axis=0)
        xs = [t + 1 for t in trials]  # 1-indexed trial labels
        ax.errorbar(xs, m, yerr=s, color="black", marker="o", lw=1.5, capsize=3, zorder=2)
        # individual seed points
        for row in mat:
            ax.scatter(xs, row, color="0.6", s=18, zorder=3)
        ax.set_xlabel("trial")
        ax.set_xticks(xs)
        ax.set_title(f"k={k}")
        if i == 0:
            ax.set_ylabel("success rate")
    fig.tight_layout()
    fig.savefig(OUT / "per_trial_score_by_k.jpg", dpi=140, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    print("saved per_trial_score_by_k.jpg")


def main():
    runs = load_runs()
    for k in sorted(runs):
        print(f"k={k}: {len(runs[k])} runs (seeds {[r['seed'] for r in runs[k]]})")
    plot_ada_delta_vs_k(runs)
    plot_per_trial_by_k(runs)


if __name__ == "__main__":
    main()
