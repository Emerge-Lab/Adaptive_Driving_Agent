"""F1-F4 per-map adaptation analysis on the 540-map x 20-rollout re-eval.

Reads outputs/eval540/per_map_k{k}_seed{s}_{wid}.csv (cols: map_id, t0..t_{k-1},
ada_delta_last_minus_0). All runs eval the SAME 539 nuplan_hard scenes, so
map_id aligns across k and seed.

F1  per-map Δ distribution per k (violin, seed-averaged) + typical noise band.
F2  conditional outcomes among maps WITH ROOM (t0<0.8): improved/flat/regressed
    by a per-map binomial z-test (|z|>2 at 60 rollouts), faceted by k.
F3  joint (t0 -> t_last) 2D histogram per k (above diagonal = adapted).
F4  seed consistency: per-map Δ correlation across the 3 seed pairs, per k
    (bar of mean Pearson r + faceted scatter). The real-vs-noise test.
"""
import csv
import glob
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("outputs/eval540")
OUT = SRC
N_ROLLOUTS = 20            # per seed
ROOM_MAX = 0.8            # t0 < ROOM_MAX => had room to improve
KS = [2, 3, 4, 5, 6]


def load():
    """Return data[k][seed] = dict map_id -> (t0, t_last, delta), and per-trial."""
    data = defaultdict(dict)
    for f in sorted(glob.glob(str(SRC / "per_map_k*_seed*.csv"))):
        m = re.search(r"per_map_k(\d+)_seed(\d+)_", f)
        k, seed = int(m.group(1)), int(m.group(2))
        rows = list(csv.DictReader(open(f)))
        tcols = sorted([c for c in rows[0] if c.startswith("t") and c[1:].isdigit()],
                       key=lambda c: int(c[1:]))
        d = {}
        for r in rows:
            mid = int(float(r["map_id"]))
            t0 = float(r[tcols[0]]); tl = float(r[tcols[-1]])
            d[mid] = (t0, tl, tl - t0)
        data[k][seed] = d
    return data


def seed_avg(data, k):
    """Common-map seed-averaged arrays for k. Returns map_ids, t0, t_last, delta."""
    seeds = sorted(data[k])
    common = set.intersection(*[set(data[k][s]) for s in seeds])
    mids = sorted(common)
    t0 = np.array([np.mean([data[k][s][m][0] for s in seeds]) for m in mids])
    tl = np.array([np.mean([data[k][s][m][1] for s in seeds]) for m in mids])
    return np.array(mids), t0, tl, tl - t0


def ztest_delta(t0, tl, n_eff):
    """Per-map z for delta under binomial null. n_eff = total rollouts behind each rate."""
    var = (t0 * (1 - t0) + tl * (1 - tl)) / n_eff
    se = np.sqrt(np.maximum(var, 1e-9))
    return (tl - t0) / se


# ---------- F1 ----------
def f1(data):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    parts_data = []
    means = []
    for k in KS:
        _, t0, tl, d = seed_avg(data, k)
        parts_data.append(d)
        means.append(d.mean())
    vp = ax.violinplot(parts_data, positions=KS, widths=0.7, showmeans=False, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor("0.6"); b.set_alpha(0.5)
    ax.plot(KS, means, "k-o", lw=1.5, zorder=3, label="mean")
    # typical noise band (2 sigma at p=0.85, 60 rollouts)
    sig = 2 * np.sqrt(2 * 0.85 * 0.15 / (3 * N_ROLLOUTS))
    ax.axhspan(-sig, sig, color="tab:blue", alpha=0.10, label=f"±2σ noise (~±{sig:.02f})")
    ax.axhline(0, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("k"); ax.set_ylabel("per-map ada_delta (t_last − t0)")
    ax.set_xticks(KS); ax.set_title("F1: per-map Δ distribution by k (seed-averaged, 60 rollouts)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "f1_delta_distribution_by_k.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved f1_delta_distribution_by_k.jpg")


# ---------- F2 ----------
def f2(data):
    imp, flat, regr, tot = [], [], [], []
    for k in KS:
        _, t0, tl, d = seed_avg(data, k)
        room = t0 < ROOM_MAX
        z = ztest_delta(t0[room], tl[room], 3 * N_ROLLOUTS)
        imp.append(int((z > 2).sum())); regr.append(int((z < -2).sum()))
        flat.append(int(((z >= -2) & (z <= 2)).sum())); tot.append(int(room.sum()))
    imp, flat, regr, tot = map(np.array, (imp, flat, regr, tot))
    x = np.arange(len(KS)); w = 0.6
    fig, ax = plt.subplots(figsize=(7, 4.8))
    pi = 100 * imp / tot; pf = 100 * flat / tot; pr = 100 * regr / tot
    ax.bar(x, pi, w, label="improved (z>2)", color="tab:green")
    ax.bar(x, pf, w, bottom=pi, label="flat", color="0.8")
    ax.bar(x, pr, w, bottom=pi + pf, label="regressed (z<−2)", color="tab:red")
    for i in range(len(KS)):
        ax.text(x[i], 101, f"{imp[i]}↑/{regr[i]}↓\nof {tot[i]}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_ylabel("% of maps with room (t0<0.8)"); ax.set_ylim(0, 112)
    ax.set_title("F2: significant per-map outcomes among maps with room (60-rollout z-test)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "f2_conditional_outcomes_by_k.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved f2_conditional_outcomes_by_k.jpg")
    for i, k in enumerate(KS):
        print(f"  k={k}: of {tot[i]} w/room — improved {imp[i]} flat {flat[i]} regressed {regr[i]}")


# ---------- F3 ----------
def f3(data):
    fig, axes = plt.subplots(1, len(KS), figsize=(3.4 * len(KS), 3.6), squeeze=False)
    edges = np.linspace(-0.025, 1.025, 22)  # 21 bins (0.05 res)
    for i, k in enumerate(KS):
        _, t0, tl, d = seed_avg(data, k)
        ax = axes[0, i]
        H, _, _ = np.histogram2d(t0, tl, bins=[edges, edges])
        ax.imshow(np.log1p(H.T), origin="lower", extent=[-0.025, 1.025, -0.025, 1.025],
                  cmap="viridis", aspect="auto")
        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.7)
        n_ad = int((tl > t0 + 0.1).sum()); n_re = int((tl < t0 - 0.1).sum())
        ax.set_title(f"k={k}  ↑{n_ad}/↓{n_re}", fontsize=10)
        ax.set_xlabel("t0");
        if i == 0: ax.set_ylabel("t_last")
    fig.suptitle("F3: joint (t0 → t_last) per map, seed-averaged (above diagonal = adapted)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "f3_joint_t0_tlast_by_k.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved f3_joint_t0_tlast_by_k.jpg")


# ---------- F4 ----------
def f4(data):
    # per-map delta per seed, common maps per k
    corr_mean, corr_all = [], {}
    for k in KS:
        seeds = sorted(data[k])
        common = sorted(set.intersection(*[set(data[k][s]) for s in seeds]))
        delt = {s: np.array([data[k][s][m][2] for m in common]) for s in seeds}
        rs = []
        for a, b in combinations(seeds, 2):
            r = np.corrcoef(delt[a], delt[b])[0, 1]
            rs.append(r)
        corr_mean.append(np.mean(rs)); corr_all[k] = (rs, delt, seeds, common)
    # F4a: bar of mean pairwise r
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(KS))
    ax.bar(x, corr_mean, 0.6, color=["tab:green" if c > 0.15 else "0.6" for c in corr_mean])
    for i, k in enumerate(KS):
        rs = corr_all[k][0]
        ax.plot([x[i]] * len(rs), rs, "k.", ms=6, zorder=3)
        ax.text(x[i], corr_mean[i] + 0.01, f"{corr_mean[i]:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_ylabel("mean cross-seed Pearson r of per-map Δ")
    ax.set_title("F4: seed consistency of per-map adaptation (high r = real, not noise)")
    fig.tight_layout(); fig.savefig(OUT / "f4_seed_consistency_corr.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved f4_seed_consistency_corr.jpg")
    # F4b: scatter seed42 vs seed43 per k
    fig, axes = plt.subplots(1, len(KS), figsize=(3.2 * len(KS), 3.4), squeeze=False, sharex=True, sharey=True)
    for i, k in enumerate(KS):
        rs, delt, seeds, common = corr_all[k]
        a, b = seeds[0], seeds[1]
        ax = axes[0, i]
        ax.scatter(delt[a], delt[b], s=6, alpha=0.4, color="0.3")
        lim = 0.6
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="0.8", lw=0.6); ax.axvline(0, color="0.8", lw=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        r = np.corrcoef(delt[a], delt[b])[0, 1]
        ax.set_title(f"k={k}  r={r:.2f}", fontsize=10)
        ax.set_xlabel(f"Δ seed{a}")
        if i == 0: ax.set_ylabel(f"Δ seed{b}")
    fig.suptitle("F4b: per-map Δ agreement across seeds (points on diagonal = consistent)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "f4_seed_consistency_scatter.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved f4_seed_consistency_scatter.jpg")
    for i, k in enumerate(KS):
        print(f"  k={k}: cross-seed r = {corr_mean[i]:.3f}  (pairs {[round(x,2) for x in corr_all[k][0]]})")


def main():
    data = load()
    for k in KS:
        print(f"k={k}: seeds {sorted(data[k])}, {len(seed_avg(data,k)[0])} common maps")
    f1(data); f2(data); f3(data); f4(data)


if __name__ == "__main__":
    main()
