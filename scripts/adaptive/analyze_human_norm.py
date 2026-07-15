"""Human-normalized per-trial adaptation scores (AdA-style aggregation).

Normalizes per-map per-trial return by the human-log return on the same map
(policy-independent: t0 of any demo-mode eval, byte-identical across wids —
see outputs/eval540_demo/human_return_per_map.csv), then aggregates with
median and 20th percentile across maps, per AdA (arXiv 2301.07608) Fig 4.

Maps with human return < H_FLOOR are excluded (ratio unstable near 0;
31/539 maps, incl. 6 where the human log itself scores <= 0).

Outputs (to outputs/eval540_norm/):
  - human_norm_scores.csv       per cell x subset x statistic x trial
  - g_human_norm_curves.jpg     2x2 figure: rows = median / 20th pct,
                                cols = headline cells
Usage: python scripts/adaptive/analyze_human_norm.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "eval540_norm"
H_FLOOR = 0.5
TRIALS = ["t0", "t1", "t2", "t3"]

CELLS = {
    "e_ub=0.10, k=4": dict(partner="2e029h15", eub=0.1,
                           wids={"qxw6c0jh": 42, "ufmegw4l": 43, "jsckmpha": 44}),
    "e_ub=0.20, k=4": dict(partner="m2ygolog", eub=0.2,
                           wids={"ftxa55g3": 42, "citbzhdc": 43, "c0k9uqhc": 44}),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(REPO / "outputs/eval540_demo/human_return_per_map.csv") \
          .set_index("map_id").human_R
    valid_h = h[h >= H_FLOOR]
    ac = pd.read_csv(REPO / "outputs/eval540_combined/all_cells.csv")

    rows = []
    curves = {}  # (cell, subset) -> dict(stat -> [4 vals], n)
    for name, c in CELLS.items():
        sel = ac[(ac.partner == c["partner"]) & (ac.entropy_ub == c["eub"]) & (ac.k == 4)]
        adaptable = sel.groupby("map_id").t0.mean().pipe(lambda p0: p0[p0 < 0.8]).index
        R = pd.concat([
            pd.read_csv(REPO / f"outputs/eval540_return/per_map_R_k4_seed{s}_{w}.csv")
              .set_index("map_id")[TRIALS]
            for w, s in c["wids"].items()
        ]).groupby(level=0).mean()
        for subset, idx in [("all", valid_h.index),
                            ("adaptable", adaptable.intersection(valid_h.index))]:
            score = R.loc[R.index.intersection(idx)].div(h, axis=0).dropna()
            stats = {"median": score.median(), "p20": score.quantile(0.2)}
            curves[(name, subset)] = (stats, len(score))
            for stat, v in stats.items():
                for t in TRIALS:
                    rows.append(dict(cell=name, subset=subset, n_maps=len(score),
                                     statistic=stat, trial=t, score=float(v[t])))
    pd.DataFrame(rows).to_csv(OUT / "human_norm_scores.csv", index=False)

    # ---- figure: rows = statistic, cols = cell ----
    C_ADAPT, C_ALL = "#1668a8", "#8a8a8a"
    x = np.arange(4)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), dpi=150, sharex=True)
    for j, cell in enumerate(CELLS):
        for i, (stat, stat_label) in enumerate([("median", "median"),
                                                ("p20", "20th percentile")]):
            ax = axs[i, j]
            for subset, color in [("all", C_ALL), ("adaptable", C_ADAPT)]:
                stats, n = curves[(cell, subset)]
                y = [stats[stat][t] for t in TRIALS]
                ax.plot(x, y, "-o", color=color, lw=2, ms=6,
                        label=f"{subset} maps (n={n})")
            ax.axhline(1.0, color="black", ls=":", lw=1.2,
                       label="human (= 1)" if (i == 0 and j == 0) else None)
            ax.set_title(f"{cell} — {stat_label}", fontsize=11)
            ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xticks(x, [f"trial {k}" for k in range(4)])
            if j == 0:
                ax.set_ylabel("return / human return")
            if i == 0 and j == 0:
                ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Human-normalized per-trial score (AdA-style aggregation)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "g_human_norm_curves.jpg", bbox_inches="tight")
    print(f"wrote {OUT}/human_norm_scores.csv and g_human_norm_curves.jpg")


if __name__ == "__main__":
    main()
