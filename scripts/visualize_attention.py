"""Render heatmaps from a probe_attention.py output.

Produces, for each (layer, head):
  - A (T, horizon) heatmap with scenario boundaries marked.
  - A 1D plot of cross-scenario attention mass over time.
Plus an overall summary plot aggregating layers × heads.

Usage:
  python scripts/visualize_attention.py /tmp/probe_attention.npz \
      [--out-dir /tmp/probe_plots]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="path to probe_attention.npz")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--cmap", default="viridis")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    attn = data["attn_lhth"]  # (L, H, T, horizon)
    L, H, T, horizon = attn.shape
    k = int(data["k"])
    scen_len = int(data["scen_len"])
    map_rand = bool(data["map_rand"])
    ckpt = str(data["checkpoint"])

    out_dir = args.out_dir or args.npz.replace(".npz", "_plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"saving plots → {out_dir}")
    print(f"L={L} H={H} T={T} horizon={horizon} k={k} scen_len={scen_len} "
          f"map_rand={map_rand}")

    # Per (layer, head) heatmap.
    boundaries = [scen_len * i for i in range(1, k)]
    for li in range(L):
        for h in range(H):
            mat = attn[li, h]  # (T, horizon)
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(
                mat, aspect="auto", origin="lower",
                cmap=args.cmap, interpolation="nearest", vmin=0,
            )
            for b in boundaries:
                ax.axhline(b - 0.5, color="red", linewidth=1.0, alpha=0.7,
                           label="query scenario boundary" if b == boundaries[0] else None)
                ax.axvline(b - 0.5, color="cyan", linewidth=1.0, alpha=0.7,
                           label="key scenario boundary" if b == boundaries[0] else None)
            ax.set_xlabel("key position (cache slot)")
            ax.set_ylabel("query step (time)")
            ax.set_title(f"layer={li} head={h}  attention(query, key)\n"
                         f"k={k} scen_len={scen_len} map_rand={map_rand}")
            ax.legend(loc="upper right", fontsize=8)
            plt.colorbar(im, ax=ax, label="softmax weight")
            fname = os.path.join(out_dir, f"attn_l{li}_h{h}.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=120)
            plt.close(fig)

    # Cross-scenario mass per (layer, head) over time.
    # For each query step t, fraction of softmax mass on slots < (t // scen_len) * scen_len
    fig, axes = plt.subplots(L, H, figsize=(4 * H, 3 * L), squeeze=False, sharex=True, sharey=True)
    times = np.arange(T)
    cur_scen = times // scen_len  # (T,) which scenario this query is in
    for li in range(L):
        for h in range(H):
            mat = attn[li, h]  # (T, horizon)
            cross_mass = np.zeros(T, dtype=np.float32)
            for t in range(T):
                past_end = cur_scen[t] * scen_len  # all slots written in earlier scenarios
                if past_end > 0:
                    cross_mass[t] = mat[t, :past_end].sum()
            ax = axes[li, h]
            ax.plot(times, cross_mass, linewidth=0.8)
            for b in boundaries:
                ax.axvline(b, color="red", linewidth=0.5, alpha=0.5)
            ax.set_title(f"L{li} H{h}", fontsize=9)
            ax.set_ylim(-0.02, 1.02)
            if li == L - 1:
                ax.set_xlabel("query step")
            if h == 0:
                ax.set_ylabel("attn mass on past scenarios")
    fig.suptitle(f"Cross-scenario attention mass over time\n"
                 f"k={k} scen_len={scen_len} map_rand={map_rand}", y=1.0)
    fig.tight_layout()
    out = os.path.join(out_dir, "cross_scenario_mass_grid.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → {out}")

    # Overall summary: mean cross-scenario mass per (layer, head), printed.
    print()
    print("=== per (layer, head) mean cross-scenario attention mass ===")
    print(f"(across all query steps in scenarios 1..{k-1})")
    for li in range(L):
        for h in range(H):
            mat = attn[li, h]  # (T, horizon)
            cross_per_t = np.zeros(T, dtype=np.float32)
            for t in range(T):
                past_end = cur_scen[t] * scen_len
                if past_end > 0:
                    cross_per_t[t] = mat[t, :past_end].sum()
            # average over all steps that have past
            mask = cur_scen > 0
            mean = cross_per_t[mask].mean() if mask.any() else 0.0
            print(f"  L{li} H{h}: {mean:.4f}")

    print()
    print(f"all plots in: {out_dir}")


if __name__ == "__main__":
    main()
