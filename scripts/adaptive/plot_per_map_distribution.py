"""Honest per-map adaptation views (handles ceiling + rollout noise).

1. 2D histogram of (t0, t_last) per k on an 11x11 grid. Above the diagonal =
   adapted, below = regressed, blob at (1,1) = already maxed (no room).
2. Conditional bar chart: among maps WITH ROOM (t0 < ROOM_MAX), count
   improved (delta > THRESH) / regressed (delta < -THRESH) / flat. THRESH=0.3
   (~1.5 sigma at 10 rollouts) filters sampling noise.

Reads outputs/per_map_adaptation/per_map_k{k}_seed{s}_{rid}.csv.
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
EXCLUDE = {"33od5iqw"}   # degraded k4 seed43 (never converged)
# Success rates are quantized to 0.1 (10 rollouts), so deltas are multiples of
# 0.1. Put the threshold BETWEEN quantization levels (0.25) so "delta >= 0.3"
# counts as real and "delta <= 0.2" (noise, ~0.22 std) is excluded — and so
# `tl - t0 > THRESH` and `tl > t0 + THRESH` agree (0.3 sits exactly on a level
# and float-rounds inconsistently).
THRESH = 0.25
ROOM_MAX = 0.8         # t0 < ROOM_MAX = had room to improve


def load_t0_tlast():
    """Return {k: (t0_pooled, tlast_pooled)} numpy arrays across seeds."""
    by_k_t0 = defaultdict(list)
    by_k_tl = defaultdict(list)
    for f in sorted(SRC.glob("per_map_k*_seed*.csv")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        k, rid = int(m.group(1)), m.group(3)
        if rid in EXCLUDE:
            continue
        with open(f) as fh:
            r = csv.DictReader(fh)
            cols = r.fieldnames
            rows = list(r)
        tcols = sorted([c for c in cols if c.startswith("t") and c[1:].isdigit()],
                       key=lambda c: int(c[1:]))
        t0c, tlc = tcols[0], tcols[-1]
        by_k_t0[k].extend(float(x[t0c]) for x in rows)
        by_k_tl[k].extend(float(x[tlc]) for x in rows)
    return {k: (np.array(by_k_t0[k]), np.array(by_k_tl[k])) for k in by_k_t0}


def plot_joint(data):
    ks = sorted(data)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.2), squeeze=False)
    edges = np.linspace(-0.05, 1.05, 12)  # 11 bins centered on 0,0.1,...,1.0
    for i, k in enumerate(ks):
        t0, tl = data[k]
        ax = axes[0, i]
        H, _, _ = np.histogram2d(t0, tl, bins=[edges, edges])
        # log-ish color so the (1,1) blob doesn't wash everything out
        im = ax.imshow(np.log1p(H.T), origin="lower", extent=[-0.05, 1.05, -0.05, 1.05],
                       cmap="viridis", aspect="auto")
        ax.plot([0, 1], [0, 1], color="white", lw=1, ls="--", alpha=0.7)
        n_adapt = int((tl > t0 + THRESH).sum())
        n_regr = int((tl < t0 - THRESH).sum())
        ax.set_title(f"k={k}   ↑adapt {n_adapt} / ↓regr {n_regr}")
        ax.set_xlabel("trial-0 success")
        if i == 0:
            ax.set_ylabel("trial-last success")
        plt.colorbar(im, ax=ax, label="log(1+#maps)")
    fig.tight_layout()
    fig.savefig(OUT / "joint_t0_tlast_by_k.jpg", dpi=140, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    print("saved joint_t0_tlast_by_k.jpg")


def plot_conditional_bars(data):
    ks = sorted(data)
    improved, regressed, flat = [], [], []
    for k in ks:
        t0, tl = data[k]
        room = t0 < ROOM_MAX
        d = (tl - t0)[room]
        improved.append(int((d > THRESH).sum()))
        regressed.append(int((d < -THRESH).sum()))
        flat.append(int((np.abs(d) <= THRESH).sum()))
    improved = np.array(improved); regressed = np.array(regressed); flat = np.array(flat)
    x = np.arange(len(ks)); w = 0.6
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x, improved, w, label=f"improved (Δ>{THRESH})", color="tab:green")
    ax.bar(x, flat, w, bottom=improved, label=f"flat (|Δ|≤{THRESH})", color="0.8")
    ax.bar(x, regressed, w, bottom=improved + flat, label=f"regressed (Δ<−{THRESH})", color="tab:red")
    for i in range(len(ks)):
        tot = improved[i] + flat[i] + regressed[i]
        ax.text(x[i], tot + 2, f"{improved[i]}↑/{regressed[i]}↓\nof {tot}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel(f"# maps with room (t0<{ROOM_MAX})")
    ax.set_title(f"adaptation among maps with room (10-rollout, {THRESH} thresh)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "conditional_adaptation_by_k.jpg", dpi=140, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    print("saved conditional_adaptation_by_k.jpg")
    # also print the numbers
    for i, k in enumerate(ks):
        print(f"  k={k}: of {improved[i]+flat[i]+regressed[i]} maps w/ room — "
              f"improved {improved[i]}, flat {flat[i]}, regressed {regressed[i]}")


def main():
    data = load_t0_tlast()
    for k in sorted(data):
        print(f"k={k}: {len(data[k][0])} pooled map-evals")
    plot_joint(data)
    plot_conditional_bars(data)


if __name__ == "__main__":
    main()
