"""G1-G4 partner-aware adaptation analysis: co-player entropy_ub x k.

Joins the full 60-cell eval (4 co-player entropy levels x 5 k x 3 seeds) into a
single adaptation picture. entropy_ub is the co-player conditioning knob (how
stochastic the partner the ego trained against was); k is the number of
adaptation trials the ego gets per map at eval. ada_delta = t_last - t0 is the
per-map change in success rate from first to last trial.

Sources (all eval the SAME 539 nuplan_hard scenes, so map_id aligns everywhere):
  outputs/eval540_grid/per_map_k{k}_seed{s}_{wid}.csv   45 cells, 3 partners
      -> entropy via scripts/adaptive/final_runs_manifest.csv (wid -> entropy_ub)
  outputs/eval540/per_map_k{k}_seed{s}_{wid}.csv         15 cells, partner 2e029h15
      -> entropy_ub = 0.10 (the column missing from the manifest)

Figures (-> outputs/eval540_entropy_k/):
  G1  entropy x k mean per-map Delta heatmap (seed-averaged).
  G2  Delta vs entropy (one line per k) + Delta vs k (one line per entropy),
      cross-seed SEM error bars. Is there a sweet spot?
  G3  net adaptation = %improved(z>2) - %regressed(z<-2) among maps WITH ROOM
      (t0<0.8), as an entropy x k heatmap. The significant-signal version of G1.
  G4  cross-seed Pearson r of per-map Delta per cell. High r = real, not noise.
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

GRID = Path("outputs/eval540_grid")  # 45 cells, 3 partners (entropy via manifest)
SOLO = Path("outputs/eval540")  # 15 cells, partner 2e029h15 -> entropy 0.10
MANIFEST = Path("scripts/adaptive/final_runs_manifest.csv")
OUT = Path("outputs/eval540_entropy_k")
SOLO_ENTROPY = 0.10  # all SOLO wids are the 2e029h15 partner

N_ROLLOUTS = 20  # per seed
ROOM_MAX = 0.8  # t0 < ROOM_MAX => had room to improve
ENTROPIES = [0.05, 0.10, 0.20, 0.50]
KS = [2, 3, 4, 5, 6]
SEEDS = [42, 43, 44]


def wid_to_entropy():
    """wid -> entropy_ub from the manifest (45 grid wids)."""
    m = {}
    for r in csv.DictReader(open(MANIFEST)):
        m[r["wandb_id"]] = float(r["entropy_ub"])
    return m


def load():
    """data[entropy][k][seed] = {map_id: (t0, t_last, delta)}."""
    w2e = wid_to_entropy()
    data = defaultdict(lambda: defaultdict(dict))
    files = [(f, None) for f in glob.glob(str(GRID / "per_map_k*_seed*.csv"))]
    files += [(f, SOLO_ENTROPY) for f in glob.glob(str(SOLO / "per_map_k*_seed*.csv"))]
    n = 0
    for f, ent_override in files:
        m = re.search(r"per_map_k(\d+)_seed(\d+)_([0-9a-z]+)\.csv", f)
        k, seed, wid = int(m.group(1)), int(m.group(2)), m.group(3)
        ent = ent_override if ent_override is not None else w2e[wid]
        rows = list(csv.DictReader(open(f)))
        tcols = sorted([c for c in rows[0] if c.startswith("t") and c[1:].isdigit()], key=lambda c: int(c[1:]))
        d = {}
        for r in rows:
            mid = int(float(r["map_id"]))
            t0 = float(r[tcols[0]])
            tl = float(r[tcols[-1]])
            d[mid] = (t0, tl, tl - t0)
        data[ent][k][seed] = d
        n += 1
    print(f"loaded {n} cells")
    return data


def seed_avg(data, ent, k):
    """Common-map seed-averaged arrays. Returns map_ids, t0, t_last, delta."""
    seeds = sorted(data[ent][k])
    common = set.intersection(*[set(data[ent][k][s]) for s in seeds])
    mids = sorted(common)
    t0 = np.array([np.mean([data[ent][k][s][m][0] for s in seeds]) for m in mids])
    tl = np.array([np.mean([data[ent][k][s][m][1] for s in seeds]) for m in mids])
    return np.array(mids), t0, tl, tl - t0


def ztest_delta(t0, tl, n_eff):
    """Per-map z for delta under binomial null. n_eff = rollouts behind each rate."""
    var = (t0 * (1 - t0) + tl * (1 - tl)) / n_eff
    se = np.sqrt(np.maximum(var, 1e-9))
    return (tl - t0) / se


def per_seed_mean_delta(data, ent, k):
    """Mean per-map delta for each seed (over that seed's own maps)."""
    out = {}
    for s in sorted(data[ent][k]):
        out[s] = float(np.mean([v[2] for v in data[ent][k][s].values()]))
    return out


def net_adapt(data, ent, k):
    """%improved(z>2) - %regressed(z<-2) among maps with room (t0<ROOM_MAX)."""
    _, t0, tl, _ = seed_avg(data, ent, k)
    room = t0 < ROOM_MAX
    if room.sum() == 0:
        return np.nan, 0
    z = ztest_delta(t0[room], tl[room], len(SEEDS) * N_ROLLOUTS)
    imp = (z > 2).sum()
    regr = (z < -2).sum()
    return 100.0 * (imp - regr) / room.sum(), int(room.sum())


def cross_seed_r(data, ent, k):
    """Mean pairwise Pearson r of per-map delta across seeds."""
    seeds = sorted(data[ent][k])
    common = sorted(set.intersection(*[set(data[ent][k][s]) for s in seeds]))
    delt = {s: np.array([data[ent][k][s][m][2] for m in common]) for s in seeds}
    rs = [np.corrcoef(delt[a], delt[b])[0, 1] for a, b in combinations(seeds, 2)]
    return float(np.mean(rs)), rs


def _grid_metric(data, fn):
    """E x K matrix of fn(data, ent, k) (scalar)."""
    M = np.full((len(ENTROPIES), len(KS)), np.nan)
    for i, e in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            if k in data[e] and len(data[e][k]) == len(SEEDS):
                M[i, j] = fn(data, e, k)
    return M


# ---------- G1 ----------
def g1(data):
    M = _grid_metric(data, lambda d, e, k: seed_avg(d, e, k)[3].mean())
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES)))
    ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k")
    ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:+.03f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else "black",
                )
    fig.colorbar(im, ax=ax, label="mean per-map Δ (t_last − t0)")
    ax.set_title("G1: mean per-map adaptation Δ by co-player entropy × k")
    fig.tight_layout()
    fig.savefig(OUT / "g1_mean_delta_heatmap.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved g1_mean_delta_heatmap.jpg")
    return M


# ---------- G2 ----------
def g2(data):
    # per cell: mean and SEM across the 3 per-seed mean-deltas
    mean = np.full((len(ENTROPIES), len(KS)), np.nan)
    sem = np.full((len(ENTROPIES), len(KS)), np.nan)
    for i, e in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            if k in data[e] and len(data[e][k]) == len(SEEDS):
                vals = np.array(list(per_seed_mean_delta(data, e, k).values()))
                mean[i, j] = vals.mean()
                sem[i, j] = vals.std(ddof=1) / np.sqrt(len(vals))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    # left: Δ vs entropy, one line per k
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(KS)))
    for j, k in enumerate(KS):
        axes[0].errorbar(
            ENTROPIES, mean[:, j], yerr=sem[:, j], marker="o", lw=1.5, capsize=3, color=cmap[j], label=f"k={k}"
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks(ENTROPIES)
    axes[0].set_xticklabels([f"{e:.2f}" for e in ENTROPIES])
    axes[0].axhline(0, color="0.7", ls=":", lw=0.8)
    axes[0].set_xlabel("co-player entropy_ub (log)")
    axes[0].set_ylabel("mean per-map Δ")
    axes[0].set_title("Δ vs co-player entropy (line per k)")
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    # right: Δ vs k, one line per entropy
    cmap2 = plt.cm.plasma(np.linspace(0.1, 0.85, len(ENTROPIES)))
    for i, e in enumerate(ENTROPIES):
        axes[1].errorbar(
            KS, mean[i, :], yerr=sem[i, :], marker="s", lw=1.5, capsize=3, color=cmap2[i], label=f"entropy={e:.2f}"
        )
    axes[1].axhline(0, color="0.7", ls=":", lw=0.8)
    axes[1].set_xticks(KS)
    axes[1].set_xlabel("adaptation trials k")
    axes[1].set_ylabel("mean per-map Δ")
    axes[1].set_title("Δ vs k (line per co-player entropy)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("G2: adaptation Δ across the entropy × k grid (error bars = cross-seed SEM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "g2_delta_lines.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved g2_delta_lines.jpg")
    return mean, sem


# ---------- G3 ----------
def g3(data):
    M = _grid_metric(data, lambda d, e, k: net_adapt(d, e, k)[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES)))
    ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k")
    ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else "black",
                )
    fig.colorbar(im, ax=ax, label="net adaptation %  (improved − regressed)")
    ax.set_title("G3: net significant adaptation among maps with room (t0<0.8)")
    fig.tight_layout()
    fig.savefig(OUT / "g3_net_adapt_heatmap.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved g3_net_adapt_heatmap.jpg")
    return M


# ---------- G4 ----------
def g4(data):
    M = _grid_metric(data, lambda d, e, k: cross_seed_r(d, e, k)[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.3, np.nanmax(M)), aspect="auto")
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES)))
    ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k")
    ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if M[i, j] < 0.5 * np.nanmax(M) else "black",
                )
    fig.colorbar(im, ax=ax, label="mean cross-seed Pearson r of per-map Δ")
    ax.set_title("G4: seed consistency of per-map adaptation (high = real signal)")
    fig.tight_layout()
    fig.savefig(OUT / "g4_seed_consistency_heatmap.jpg", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved g4_seed_consistency_heatmap.jpg")
    return M


def summary(data, g1M, g3M, g4M):
    print("\n=== entropy × k summary (seed-averaged) ===")
    print(f"{'ent':>5} {'k':>2} {'maps':>5} {'meanΔ':>8} {'netAdapt%':>9} {'xseed_r':>8}")
    for i, e in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            if k in data[e] and len(data[e][k]) == len(SEEDS):
                mids = seed_avg(data, e, k)[0]
                na, nroom = net_adapt(data, e, k)
                print(f"{e:>5.2f} {k:>2} {len(mids):>5} {g1M[i, j]:>+8.04f} {g3M[i, j]:>+8.1f}% {g4M[i, j]:>8.03f}")
    # best cell by each metric
    bi = np.unravel_index(np.nanargmax(g1M), g1M.shape)
    print(f"\nbest mean Δ:      entropy={ENTROPIES[bi[0]]:.2f} k={KS[bi[1]]} ({g1M[bi]:+.4f})")
    bi = np.unravel_index(np.nanargmax(g3M), g3M.shape)
    print(f"best net adapt %: entropy={ENTROPIES[bi[0]]:.2f} k={KS[bi[1]]} ({g3M[bi]:+.1f}%)")
    bi = np.unravel_index(np.nanargmax(g4M), g4M.shape)
    print(f"best seed r:      entropy={ENTROPIES[bi[0]]:.2f} k={KS[bi[1]]} ({g4M[bi]:.3f})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    # coverage report
    for e in ENTROPIES:
        ks = {k: sorted(data[e][k]) for k in KS if k in data[e]}
        print(f"entropy={e:.2f}: " + " ".join(f"k{k}{v}" for k, v in ks.items()))
    g1M = g1(data)
    g2(data)
    g3M = g3(data)
    g4M = g4(data)
    summary(data, g1M, g3M, g4M)


if __name__ == "__main__":
    main()
