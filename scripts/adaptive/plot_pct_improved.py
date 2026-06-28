"""Per-cell fraction of ADAPTABLE maps (p_0 < 0.8) that IMPROVED across trials.

"Improved" = cross-seed mean ΔR > 0 (loose) and > 0.5 (meaningful: ~ one fewer
crash OR half a goal). Counts and fractions are over the common-map intersection
across the 3 seeds, restricted to the p_0 < 0.8 filter.

Inputs: outputs/eval540_return/all_cells_R.csv (return) +
        outputs/eval540_combined/all_cells.csv (p_0 selector).

Outputs in outputs/eval540_return/:
  g_pct_improved_room.jpg  -- 2-panel heatmap: %improved at thresh 0 and 0.5
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_R = Path("outputs/eval540_return/all_cells_R.csv")
IN_P = Path("outputs/eval540_combined/all_cells.csv")
OUT_DIR = Path("outputs/eval540_return")

ENTROPIES = [0.05, 0.10, 0.20, 0.50]
KS = [2, 3, 4, 5, 6]
SEEDS = [42, 43, 44]


def load_return():
    data = defaultdict(lambda: defaultdict(dict))
    with open(IN_R) as f:
        for r in csv.DictReader(f):
            ent = float(r["entropy_ub"]); k = int(r["k"]); seed = int(r["seed"])
            mid = int(r["map_id"])
            ts = tuple(float(r[f"t{i}"]) for i in range(k))
            data[ent][k].setdefault(seed, {})[mid] = ts
    return data


def load_p0_keep(threshold=0.8):
    p0 = defaultdict(lambda: defaultdict(dict))
    with open(IN_P) as f:
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


def pct_improved(data, keep, thresh):
    """Per cell: (pct improved, n_adaptable_common, n_improved)."""
    pct = np.full((len(ENTROPIES), len(KS)), np.nan)
    Ns = np.zeros_like(pct, dtype=int)
    Nimp = np.zeros_like(pct, dtype=int)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            allowed = keep.get((ent, k), set())
            per_seed_delta = {}
            for seed in SEEDS:
                rows = data[ent][k].get(seed, {})
                if not rows:
                    continue
                sel = sorted(m for m in rows if m in allowed)
                if not sel:
                    continue
                arr = np.array([rows[m] for m in sel])
                per_seed_delta[seed] = (sel, arr[:, -1] - arr[:, 0])
            if len(per_seed_delta) != len(SEEDS):
                continue
            seeds_sorted = sorted(per_seed_delta)
            sets = [set(per_seed_delta[s][0]) for s in seeds_sorted]
            common = sorted(set.intersection(*sets))
            if not common:
                continue
            stack = np.stack([
                np.array([per_seed_delta[s][1][per_seed_delta[s][0].index(m)] for m in common])
                for s in seeds_sorted
            ])
            mean_per_map = stack.mean(axis=0)
            n_imp = int((mean_per_map > thresh).sum())
            Ns[i, j] = len(common)
            Nimp[i, j] = n_imp
            pct[i, j] = 100.0 * n_imp / len(common)
    return pct, Ns, Nimp


def plot_two_panel(pct0, N, Nimp0, pct1, Nimp1, thresh1, fname):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    panels = [
        (axes[0], pct0, Nimp0, 0.0,    "Improved at any level (mean ΔR > 0)"),
        (axes[1], pct1, Nimp1, thresh1, f"Meaningfully improved (mean ΔR > {thresh1})"),
    ]
    vmax = 100.0
    for ax, M, Ni, _thresh, title in panels:
        im = ax.imshow(M, cmap="Greens", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
        ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
        ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
        for i in range(len(ENTROPIES)):
            for j in range(len(KS)):
                if np.isnan(M[i, j]):
                    continue
                txt = f"{M[i,j]:.0f}%\n{Ni[i,j]}/{N[i,j]}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 55 else "black")
        fig.colorbar(im, ax=ax, label="% maps improved")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Fraction of adaptable maps (p_0 < 0.8) with cross-seed mean ΔR > threshold",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_return()
    keep = load_p0_keep(threshold=0.8)
    pct0, N, Nimp0 = pct_improved(data, keep, thresh=0.0)
    pct1, _, Nimp1 = pct_improved(data, keep, thresh=0.5)
    plot_two_panel(pct0, N, Nimp0, pct1, Nimp1, 0.5, "g_pct_improved_room.jpg")

    print("\n=== % adaptable maps with cross-seed mean ΔR > 0 ===")
    print(f"{'ent':>5} {'k':>2} {'n':>4} {'n_imp':>5} {'pct':>6} | "
          f"{'n_imp>0.5':>9} {'pct>0.5':>8}")
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            if np.isnan(pct0[i, j]):
                continue
            print(f"{ent:>5.2f} {k:>2} {N[i,j]:>4} {Nimp0[i,j]:>5} "
                  f"{pct0[i,j]:>5.1f}% | {Nimp1[i,j]:>9} {pct1[i,j]:>7.1f}%")


if __name__ == "__main__":
    main()
