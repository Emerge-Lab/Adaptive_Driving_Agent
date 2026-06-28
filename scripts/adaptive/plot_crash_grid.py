"""Per-trial CRASH-STEP rate plots from the existing per_map_crash_*.csv files.

The crash CSVs store summed crash reward (= -0.5 x #crash_steps) per (map, trial).
Negating and dividing by 0.5 gives the per-trial crash-step count, averaged
across 20 rollouts. This is the cleanest adaptation signal: crashes are
discrete events, untouched by lane shaping, and the threshold (r <= -0.45) is
robust because per-step crash penalty is always exactly -0.5 +/- tiny lane.

Mirrors plot_return_grid.py: filters to adaptable maps (cross-seed mean p_0 < 0.8).

Outputs in outputs/eval540_return/:
  g_crash_per_trial_mean_std_room.jpg  -- 4x5 cross-seed mean +/- std curves
  g_crash_mean_delta_heatmap_room.jpg  -- mean Delta(crash-steps) per cell
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
OUT_DIR = Path("outputs/eval540_return")
MANIFEST = Path("scripts/adaptive/final_runs_manifest.csv")
SOLO_ENTROPY = 0.10
CRASH_PENALTY = 0.5  # magnitude of -0.5 per crash step

ENTROPIES = [0.05, 0.10, 0.20, 0.50]
KS = [2, 3, 4, 5, 6]
SEEDS = [42, 43, 44]


def load_crash_steps():
    """data[ent][k][seed][map_id] = (c0, c1, ..., c_{k-1})  in crash-step counts."""
    pat = re.compile(r"per_map_crash_k(\d+)_seed(\d+)_([0-9a-z]+)\.csv$")
    manifest = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            manifest[r["wandb_id"]] = float(r["entropy_ub"])
    data = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(str(IN_DIR / "per_map_crash_k*_seed*.csv"))):
        m = pat.search(Path(path).name)
        if m is None:
            continue
        k, seed, wid = int(m.group(1)), int(m.group(2)), m.group(3)
        ent = manifest.get(wid, SOLO_ENTROPY)
        with open(path) as fh:
            for r in csv.DictReader(fh):
                mid = int(float(r["map_id"]))
                # crash CSV holds sum of -0.5 per crash step -> negate & /0.5
                ts = tuple(-float(r[f"t{i}"]) / CRASH_PENALTY for i in range(k))
                if seed not in data[ent][k]:
                    data[ent][k][seed] = {}
                data[ent][k][seed][mid] = ts
    return data


def load_p0_filter(threshold=0.8):
    """Per (ent,k): set of map_ids whose cross-seed mean p_0 < threshold."""
    p0 = defaultdict(lambda: defaultdict(dict))
    with open(SUCCESS_CSV) as f:
        for r in csv.DictReader(f):
            ent = float(r["entropy_ub"])
            k = int(r["k"])
            seed = int(r["seed"])
            mid = int(r["map_id"])
            p0[(ent, k)][seed][mid] = float(r["t0"])
    keep = {}
    for cell, by_seed in p0.items():
        seeds = list(by_seed)
        common = sorted(set.intersection(*(set(by_seed[s]) for s in seeds)))
        mean_p0 = {m: float(np.mean([by_seed[s][m] for s in seeds])) for m in common}
        keep[cell] = {m for m, p in mean_p0.items() if p < threshold}
    return keep


def plot_per_trial_mean_std(data, keep, fname):
    fig, axes = plt.subplots(len(ENTROPIES), len(KS),
                             figsize=(15, 10), sharey=True)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            ax = axes[i, j]
            allowed = keep.get((ent, k), set())
            per_seed_mean = {}
            for seed in SEEDS:
                rows = data[ent][k].get(seed, {})
                if not rows:
                    continue
                sel = [rows[m] for m in rows if m in allowed]
                if not sel:
                    continue
                per_seed_mean[seed] = np.array(sel).mean(axis=0)
            if len(per_seed_mean) >= 2:
                stack = np.stack([per_seed_mean[s] for s in sorted(per_seed_mean)])
                mean = stack.mean(axis=0)
                std = stack.std(axis=0, ddof=1)
                ax.fill_between(range(k), mean - std, mean + std,
                                color="tab:purple", alpha=0.22, label="+/-1 std")
                ax.plot(range(k), mean, marker="o", lw=2.0, color="tab:purple",
                        label="cross-seed mean")
            ax.set_title(f"ent={ent:.2f}, k={k} (n={len(allowed)})", fontsize=9)
            ax.set_xticks(range(k))
            ax.grid(True, alpha=0.3)
            if i == len(ENTROPIES) - 1:
                ax.set_xlabel("trial index")
            if j == 0:
                ax.set_ylabel(f"ent={ent:.2f}\nmean #crash-steps")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle(
        "Per-trial mean #crash-steps on adaptable maps (p_0 < 0.8) -- "
        "cross-seed mean +/- 1 std. LOWER is BETTER.",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_mean_delta_heatmap(data, keep, fname):
    """Mean per-map Delta(crash-steps) = mean_last - mean_0. Negative = fewer crashes."""
    M = np.full((len(ENTROPIES), len(KS)), np.nan)
    Ns = np.full((len(ENTROPIES), len(KS)), 0, dtype=int)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            per_seed_delta = {}
            for seed in SEEDS:
                rows = data[ent][k].get(seed, {})
                if not rows:
                    continue
                allowed = keep.get((ent, k), set())
                sel_mids = sorted(m for m in rows if m in allowed)
                if not sel_mids:
                    continue
                arr = np.array([rows[m] for m in sel_mids])
                per_seed_delta[seed] = (sel_mids, arr[:, -1] - arr[:, 0])
            if len(per_seed_delta) != len(SEEDS):
                continue
            seeds_sorted = sorted(per_seed_delta)
            sets = [set(per_seed_delta[s][0]) for s in seeds_sorted]
            common = sorted(set.intersection(*sets))
            stack = np.stack([
                np.array([per_seed_delta[s][1][per_seed_delta[s][0].index(m)] for m in common])
                for s in seeds_sorted
            ])
            M[i, j] = float(stack.mean(axis=0).mean())
            Ns[i, j] = len(common)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    vmax = float(np.nanmax(np.abs(M)))
    # blue = fewer crashes (good), red = more crashes (bad)
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.02f}\nn={Ns[i,j]}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, label="mean per-map Delta(#crash-steps), last - 0")
    ax.set_title(
        "Mean per-map Delta(#crash-steps) -- adaptable maps (p_0 < 0.8). "
        "Negative = fewer crashes across trials.",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return M


def print_summary(data, keep):
    print("\n=== mean #crash-steps per trial on adaptable maps (p_0 < 0.8) ===")
    print(f"{'ent':>5} {'k':>2} {'n':>4} {'c0':>6} {'c_last':>8} {'Delta':>8}")
    for ent in ENTROPIES:
        for k in KS:
            per_seed_c0 = []
            per_seed_clast = []
            ns = []
            for seed in SEEDS:
                rows = data[ent][k].get(seed, {})
                if not rows:
                    continue
                allowed = keep.get((ent, k), set())
                sel = [rows[m] for m in rows if m in allowed]
                if not sel:
                    continue
                arr = np.array(sel)
                per_seed_c0.append(arr[:, 0].mean())
                per_seed_clast.append(arr[:, -1].mean())
                ns.append(len(sel))
            if not per_seed_c0:
                continue
            c0 = float(np.mean(per_seed_c0))
            cl = float(np.mean(per_seed_clast))
            n = int(np.mean(ns))
            print(f"{ent:>5.2f} {k:>2} {n:>4} {c0:>6.2f} {cl:>8.2f} {cl - c0:>+8.2f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_crash_steps()
    keep = load_p0_filter(threshold=0.8)
    plot_per_trial_mean_std(data, keep, "g_crash_per_trial_mean_std_room.jpg")
    plot_mean_delta_heatmap(data, keep, "g_crash_mean_delta_heatmap_room.jpg")
    print_summary(data, keep)


if __name__ == "__main__":
    main()
