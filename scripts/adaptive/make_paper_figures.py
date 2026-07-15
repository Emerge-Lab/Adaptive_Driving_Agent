"""Paper figures — one styled pipeline from outputs/ CSVs.

Figures (to outputs/paper_figs/, .pdf + .jpg):
  fig1_adaptation   human-normalized per-trial score, headline cells
                    (median + 20th pct, adaptable + all maps)
  fig2_decomposition  dR component breakdown (collision/goal/offroad/lane)
  fig3_modulators   k sweep | partner-entropy trade-off | capacity
  fig4_demo         demonstration prompting 3-way vs human
  fig5_memory       memory-reset causal control

Conventions: per-map values are means over 20 rollouts; cells are averaged
across seeds AFTER the per-map seed-mean; "adaptable" = maps with p0 < 0.8
on the cell's own binary-success selector (outputs/eval540_combined).
Usage: python scripts/adaptive/make_paper_figures.py [fig1 fig2 ...]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "paper_figs"
O = REPO / "outputs"

# ---- style ----
C_BLUE = "#1668a8"   # primary series (adaptable / intact / main cell)
C_GRAY = "#8a8a8a"   # context series (all maps / uniform)
C_RED = "#c2432f"    # ablation / collision
C_GOLD = "#c78f2d"   # secondary accent (offroad / arm B)
C_TEAL = "#2b8a78"   # tertiary accent (goal / arm A)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

CELLS = {
    "0.10": dict(partner="2e029h15", eub=0.1,
                 wids={"qxw6c0jh": 42, "ufmegw4l": 43, "jsckmpha": 44}),
    "0.20": dict(partner="m2ygolog", eub=0.2,
                 wids={"ftxa55g3": 42, "citbzhdc": 43, "c0k9uqhc": 44}),
}
T4 = ["t0", "t1", "t2", "t3"]
_ac = pd.read_csv(O / "eval540_combined/all_cells.csv")


def selector(partner, eub, k=4):
    sel = _ac[(_ac.partner == partner) & (_ac.entropy_ub == eub) & (_ac.k == k)]
    p0 = sel.groupby("map_id").t0.mean()
    return p0[p0 < 0.8].index


def cell_R(dirname, wids, k=4, component="R", trials=T4):
    """per-map seed-mean per-trial values for one cell."""
    dfs = []
    for w, s in wids.items():
        df = pd.read_csv(O / dirname / f"per_map_{component}_k{k}_seed{s}_{w}.csv")
        dfs.append(df.set_index("map_id")[trials])
    return pd.concat(dfs).groupby(level=0).mean()


def per_seed_dR(dirname, wids, maps, k=4, trials=T4):
    out = []
    for w, s in wids.items():
        df = pd.read_csv(O / dirname / f"per_map_R_k{k}_seed{s}_{w}.csv").set_index("map_id")
        sub = df.loc[df.index.intersection(maps)][trials].mean()
        out.append(float(sub[trials[-1]] - sub[trials[0]]))
    return out


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "jpg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf/.jpg")


# ---------------------------------------------------------------- fig 1
def fig1():
    h = pd.read_csv(O / "eval540_demo/human_return_per_map.csv").set_index("map_id").human_R
    valid = h[h >= 0.5].index
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (label, c) in zip(axs, CELLS.items()):
        ad = selector(c["partner"], c["eub"])
        R = cell_R("eval540_return", c["wids"])
        x = np.arange(4)
        for maps, color, name in [(valid, C_GRAY, "all maps"),
                                  (ad.intersection(valid), C_BLUE, "adaptable maps")]:
            score = R.loc[R.index.intersection(maps)].div(h, axis=0).dropna()
            ax.plot(x, score.median(), "-o", color=color, lw=2, ms=5,
                    label=f"{name} median (n={len(score)})")
            ax.plot(x, score.quantile(0.2), "--s", color=color, lw=1.4, ms=4,
                    alpha=0.8, label=f"{name} 20th pct")
        ax.axhline(1.0, color="black", ls=":", lw=1.1, label="human (= 1)")
        ax.set_title(f"$e_{{ub}}$={label}, k=4")
        ax.set_xticks(x, [f"trial {i}" for i in range(4)])
    axs[0].set_ylabel("return / human return")
    axs[0].legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig1_adaptation")


# ---------------------------------------------------------------- fig 2
def fig2():
    """Component decomposition of dR.

    Restricted to maps whose final trial completed within the horizon:
    the env-side component accumulators are snapshotted at trial-end flags,
    so a horizon-truncated final trial leaves components undefined for t3
    (~13-18/539 maps per seed, crash-heavy respawn loops). On the kept maps
    the component sum reproduces per_map_R exactly (checked to 1e-6).
    """
    comps = [("collision", C_RED), ("goal", C_TEAL), ("offroad", C_GOLD), ("lane", C_GRAY)]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.19
    ns = []
    for ci, (label, c) in enumerate(CELLS.items()):
        ad = selector(c["partner"], c["eub"])
        R = cell_R("eval540_return", c["wids"])
        D = {comp: cell_R("eval540_return", c["wids"], component=comp) for comp, _ in comps}
        csum = sum(D.values())
        complete = (R[T4] - csum[T4]).abs().max(axis=1) < 0.1
        keep = R.index[complete].intersection(ad)
        total = float((R.loc[keep].t3 - R.loc[keep].t0).mean())
        for j, (comp, color) in enumerate(comps):
            d = float((D[comp].loc[keep].t3 - D[comp].loc[keep].t0).mean())
            ax.bar(ci + (j - 1.5) * width, d, width * 0.92, color=color,
                   label=comp if ci == 0 else None)
            ax.text(ci + (j - 1.5) * width, d + (0.02 if d >= 0 else -0.05),
                    f"{d:+.2f}", ha="center", fontsize=8)
        ax.scatter([ci], [total], marker="D", color="black", zorder=5,
                   label="total ΔR" if ci == 0 else None)
        ax.text(ci + 0.02, total + 0.04, f"{total:+.2f}", fontsize=9, fontweight="bold")
        ns.append(len(keep))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([0, 1], [f"$e_{{ub}}$={k}, k=4  (n={n})" for k, n in zip(CELLS, ns)])
    ax.set_ylabel("Δ per-trial return (last − first)")
    ax.set_title("What improves: crash avoidance dominates adaptation")
    ax.legend(ncol=3, loc="upper left", framealpha=0.9)
    fig.text(0.5, -0.03, "adaptable maps whose final trial completed within the horizon",
             ha="center", fontsize=8, color="0.4")
    save(fig, "fig2_decomposition")


# ---------------------------------------------------------------- fig 3
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))

    # (a) k sweep at e_ub=0.10. k2-4 = cluster grid; k5 = standard-minibatch
    # rerun (memsafe k5/k6 cells excluded — superseded, see result log).
    ax = axs[0]
    kcells = {
        2: ("eval540_return", {"ofm0rrbm": 42, "r2wr75ay": 43, "vx2g4pcg": 44}, None),
        3: ("eval540_return", {"0qsa6hku": 42, "lkgv9a0b": 43, "pbvf72ym": 44}, None),
        4: ("eval540_return", {"qxw6c0jh": 42, "ufmegw4l": 43, "jsckmpha": 44}, None),
        5: ("eval540_k5std", {"q924lklb": 42, "qyjag2qw": 43, "15lpuj3s": 44}, "own"),
    }
    xs, means, stds = [], [], []
    for k, (dirname, wids, sel_mode) in kcells.items():
        if sel_mode == "own":  # rerun cell isn't in all_cells: own per-rollout selector
            p0 = pd.concat([pd.read_csv(O / f"{dirname}/per_rollout_success_k{k}_seed{s}_{w}.csv")
                           .groupby("map_id").t0.mean() for w, s in wids.items()],
                           axis=1).mean(axis=1)
        else:
            sel = _ac[(_ac.partner == "2e029h15") & (_ac.entropy_ub == 0.1) & (_ac.k == k)]
            p0 = sel.groupby("map_id").t0.mean()
        ad = p0[p0 < 0.8].index
        drs = per_seed_dR(dirname, wids, ad, k=k, trials=[f"t{i}" for i in range(k)])
        ax.scatter(k + np.linspace(-0.08, 0.08, 3), drs, color=C_BLUE, alpha=0.4,
                   s=26, zorder=3, clip_on=False)
        xs.append(k); means.append(np.mean(drs)); stds.append(np.std(drs))
    ax.errorbar(xs, means, yerr=stds, fmt="-o", color=C_BLUE, lw=2, ms=6,
                capsize=3, zorder=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.margins(y=0.12)
    ax.set_xticks([2, 3, 4, 5]); ax.set_xlabel("trials per episode k")
    ax.set_ylabel("ΔR (adaptable maps)"); ax.set_title("(a) trial budget")

    # (b) partner entropy: zero-shot vs adaptation
    ax = axs[1]
    ecells = [
        (0.001, "eval540_e0001", {"wyoe194j": 42, "x3vxow3u": 43, "uzfye8l1": 44}, ("2e029h15", 0.1)),
        (0.05, "eval540_return", {"9gc19bcy": 42, "bhx6zxn0": 43, "9qewt905": 44}, ("miku2puk", 0.05)),
        (0.10, "eval540_return", CELLS["0.10"]["wids"], ("2e029h15", 0.1)),
        (0.20, "eval540_return", CELLS["0.20"]["wids"], ("m2ygolog", 0.2)),
        (0.50, "eval540_return", {"m4ibxlhu": 42, "yu0vk259": 43, "7s0p1b8q": 44}, ("6rauydj2", 0.5)),
    ]
    e_x, dr_y, zs_y = [], [], []
    for eub, d, wids, (selp, sele) in ecells:
        ad = selector(selp, sele)
        R = cell_R(d, wids)
        sub = R.loc[R.index.intersection(ad)]
        e_x.append(eub)
        dr_y.append(float((sub.t3 - sub.t0).mean()))
        zs_y.append(float(R.t0.mean()))
    ax.plot(range(5), dr_y, "-o", color=C_BLUE, lw=2, label="adaptation ΔR (adaptable)")
    ax.plot(range(5), zs_y, "-s", color=C_GRAY, lw=2, label="zero-shot return (all maps)")
    ax.set_xticks(range(5), [f"{e:g}" for e in e_x])
    ax.set_xlabel("partner entropy bound $e_{ub}$")
    ax.set_title("(b) partner stochasticity"); ax.axhline(0, color="black", lw=0.8)
    ax.legend(fontsize=8, loc="upper left")

    # (c) capacity
    ax = axs[2]
    hcells = {64: {"gcknljp3": 42, "zrps018m": 43, "oxxiy4yd": 44},
              128: {"qgzr83aa": 42, "ax4hvk7w": 43, "hib8dwoo": 44},
              256: {"k29sr0rw": 42, "pv53oner": 43, "2rkkkxr3": 44}}
    hs, hmeans = [], []
    for hsize, wids in hcells.items():
        vals = []
        for w, s in wids.items():
            df = pd.read_csv(O / f"eval540_hsize/per_map_R_k4_seed{s}_{w}.csv")
            vals.append(float(df[T4].mean().mean()))
        ax.scatter([hsize] * 3, vals, color=C_BLUE, alpha=0.45, s=22)
        hs.append(hsize); hmeans.append(np.mean(vals))
    ax.plot(hs, hmeans, "-o", color=C_BLUE, lw=2)
    ax.set_xscale("log", base=2); ax.set_xticks(hs, [str(x) for x in hs])
    ax.set_xlabel("model width (hidden size)")
    ax.set_ylabel("mean eval return (all maps × trials)")
    ax.set_title("(c) capacity")
    fig.tight_layout()
    save(fig, "fig3_modulators")


# ---------------------------------------------------------------- fig 4
def fig4():
    armC = {"qgzr83aa": 42, "ax4hvk7w": 43, "hib8dwoo": 44}   # h128 baseline, no demo
    armA = armC                                                # same wids, demo-mode eval dir
    armB = {"gmfve041": 42, "gxm7jb7t": 43, "kualhrgw": 44}   # demo-trained
    h = pd.read_csv(O / "eval540_demo/human_return_per_map.csv").set_index("map_id").human_R
    ad = selector("2e029h15", 0.1)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, maps, sub in [(axs[0], None, "all maps (n=539)"),
                          (axs[1], ad, f"adaptable maps (n={len(ad)})")]:
        def curve(dirname, wids):
            R = cell_R(dirname, wids)
            if maps is not None:
                R = R.loc[R.index.intersection(maps)]
            return R[T4].mean()
        cC = curve("eval540_hsize", armC)
        cA = curve("eval540_demo", armA)
        cB = curve("eval540_demo", armB)
        hm = float(h.mean()) if maps is None else float(h.loc[h.index.intersection(maps)].mean())
        x = np.arange(4)
        ax.plot(x, cC, "-o", color=C_GRAY, lw=2, ms=5, label="no demo (baseline)")
        ax.plot(x[1:], cA[1:], "-s", color=C_TEAL, lw=2, ms=5, label="demo at eval (trial 0 = human)")
        ax.plot(x[1:], cB[1:], "-^", color=C_GOLD, lw=2, ms=5, label="demo-trained + demo at eval")
        ax.scatter([0], [cA["t0"]], marker="*", s=130, color="black", zorder=5,
                   label="human demonstration")
        ax.axhline(hm, color="black", ls=":", lw=1.1)
        ax.set_xticks(x, [f"trial {i}" for i in range(4)])
        ax.set_title(sub)
    axs[0].set_ylabel("mean per-trial return")
    axs[0].legend(fontsize=8, loc="center right", framealpha=0.9)
    fig.suptitle("A human demonstration is no substitute for own experience", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_demo")


# ---------------------------------------------------------------- fig 5
def fig5():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (label, c) in zip(axs, CELLS.items()):
        ad = selector(c["partner"], c["eub"])
        x = np.arange(4)
        for dirname, color, name in [("eval540_return", C_BLUE, "memory intact"),
                                     ("eval540_cachereset2", C_RED, "memory reset per trial")]:
            per_seed = []
            for w, s in c["wids"].items():
                df = pd.read_csv(O / dirname / f"per_map_R_k4_seed{s}_{w}.csv").set_index("map_id")
                per_seed.append(df.loc[df.index.intersection(ad)][T4].mean())
            m = pd.concat(per_seed, axis=1)
            mean, std = m.mean(axis=1), m.std(axis=1)
            dR = float(mean["t3"] - mean["t0"])
            ax.errorbar(x, mean, yerr=std, fmt="-o", color=color, lw=2, ms=5,
                        capsize=3, label=f"{name} (ΔR={dR:+.2f})")
        ax.set_title(f"$e_{{ub}}$={label}, k=4 (adaptable maps, n={len(ad)})")
        ax.set_xticks(x, [f"trial {i}" for i in range(4)])
        ax.legend(loc="center right", fontsize=8)
    axs[0].set_ylabel("mean per-trial return")
    fig.suptitle("Resetting cross-trial memory eliminates adaptation", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_memory")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for name in which:
        FIGS[name]()
