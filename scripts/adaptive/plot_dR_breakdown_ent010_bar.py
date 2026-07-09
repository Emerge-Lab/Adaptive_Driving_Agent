"""Stacked bar chart of ΔR decomposition at e_ub=0.10, one bar per k.
Each bar's segments = goal / collision / offroad / lane contributions;
labels show absolute Δ and % of ΔR_total.
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
MANIFEST = Path("scripts/adaptive/final_runs_manifest.csv")
OUT_DIR = Path("outputs/eval540_return")
SOLO_ENTROPY = 0.10

ENT_TARGET = 0.10
KS = [2, 3, 4, 5, 6]
SEEDS = [42, 43, 44]


def _load_metric(prefix):
    pat = re.compile(rf"per_map_{re.escape(prefix)}_k(\d+)_seed(\d+)_([0-9a-z]+)\.csv$")
    manifest = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            manifest[r["wandb_id"]] = float(r["entropy_ub"])
    data = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(str(IN_DIR / f"per_map_{prefix}_k*_seed*.csv"))):
        m = pat.search(Path(path).name)
        if m is None:
            continue
        k, seed, wid = int(m.group(1)), int(m.group(2)), m.group(3)
        ent = manifest.get(wid, SOLO_ENTROPY)
        with open(path) as fh:
            for r in csv.DictReader(fh):
                mid = int(float(r["map_id"]))
                ts = tuple(float(r[f"t{i}"]) for i in range(k))
                data[ent][k].setdefault(seed, {})[mid] = ts
    return data


def load_p0_keep(threshold=0.8):
    p0 = defaultdict(lambda: defaultdict(dict))
    with open(SUCCESS_CSV) as f:
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


def _delta_per_seed(data, ent, k, allowed):
    out = {}
    for seed in SEEDS:
        rows = data[ent][k].get(seed, {})
        if not rows:
            continue
        sel = sorted(m for m in rows if m in allowed)
        if not sel:
            continue
        arr = np.array([rows[m] for m in sel])
        out[seed] = (sel, arr[:, -1] - arr[:, 0])
    return out


def cell_means_for_ent(data_R, data_goal, data_collision, data_offroad, data_lane,
                       keep, ent):
    out = {}
    metrics = [("R", data_R), ("goal", data_goal), ("collision", data_collision),
               ("offroad", data_offroad), ("lane", data_lane)]
    for k in KS:
        allowed = keep.get((ent, k), set())
        per_seed_by_metric = {
            name: _delta_per_seed(d, ent, k, allowed) for name, d in metrics
        }
        if not all(per_seed_by_metric[name] for name, _ in metrics):
            continue
        all_sets = []
        ok = True
        for name, _ in metrics:
            for s in SEEDS:
                if s not in per_seed_by_metric[name]:
                    ok = False; break
                all_sets.append(set(per_seed_by_metric[name][s][0]))
            if not ok:
                break
        if not ok:
            continue
        common = sorted(set.intersection(*all_sets))
        if not common:
            continue
        def _stack(ps):
            return np.stack([
                np.array([ps[s][1][ps[s][0].index(m)] for m in common])
                for s in sorted(ps)
            ])
        means = {
            name: float(_stack(per_seed_by_metric[name]).mean(axis=0).mean())
            for name, _ in metrics
        }
        out[k] = dict(**means, n=len(common))
    return out


def plot_bar(cells, ent, fname):
    ks_present = [k for k in KS if k in cells]
    if not ks_present:
        print("no data for ent", ent)
        return

    components = ["goal", "collision", "offroad", "lane"]
    colors = {
        "goal":      "#4C9A2A",   # green
        "collision": "#C0392B",   # red
        "offroad":   "#E67E22",   # orange
        "lane":      "#2980B9",   # blue
    }
    labels = {
        "goal":      "goal (reach-rate)",
        "collision": "collision (fewer crashes)",
        "offroad":   "offroad (less off-map)",
        "lane":      "lane (per-step shaping)",
    }

    x = np.arange(len(ks_present))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))

    pos = np.zeros(len(ks_present))
    neg = np.zeros(len(ks_present))

    for comp in components:
        vals = np.array([cells[k][comp] for k in ks_present])
        # split positive vs negative contributions so a stacked bar is honest
        pos_vals = np.where(vals > 0, vals, 0.0)
        neg_vals = np.where(vals < 0, vals, 0.0)
        ax.bar(x, pos_vals, width, bottom=pos, color=colors[comp], label=labels[comp],
               edgecolor="white", linewidth=0.5)
        ax.bar(x, neg_vals, width, bottom=neg, color=colors[comp],
               edgecolor="white", linewidth=0.5)
        # annotate % of ΔR_total per component
        for i, k in enumerate(ks_present):
            R = cells[k]["R"]
            v = cells[k][comp]
            if abs(R) < 1e-6 or abs(v) < 0.005:
                continue
            pct = 100.0 * v / R
            y = pos[i] + v / 2 if v > 0 else neg[i] + v / 2
            ax.text(x[i], y, f"{v:+.02f}\n({pct:+.0f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        pos = pos + pos_vals
        neg = neg + neg_vals

    # total ΔR marker
    Rs = np.array([cells[k]["R"] for k in ks_present])
    ax.scatter(x, Rs, color="black", s=80, zorder=5, marker="D",
               label=f"ΔR_total (goal+coll+off+lane)")
    for i, R in enumerate(Rs):
        ax.text(x[i], R + 0.03, f"ΔR={R:+.02f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}\n(n={cells[k]['n']})" for k in ks_present])
    ax.set_xlabel("adaptation trials k")
    ax.set_ylabel("Δ (last trial − first trial) on adaptable maps")
    ax.set_title(f"ΔR decomposition at e_ub={ent:.2f} — where the adaptation gain comes from\n"
                 f"(adaptable maps: p_0 < 0.8)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_R         = _load_metric("R")
    data_goal      = _load_metric("goal")
    data_collision = _load_metric("collision")
    data_offroad   = _load_metric("offroad")
    data_lane      = _load_metric("lane")
    keep = load_p0_keep(threshold=0.8)
    cells = cell_means_for_ent(data_R, data_goal, data_collision, data_offroad,
                               data_lane, keep, ENT_TARGET)
    plot_bar(cells, ENT_TARGET, f"g_dR_breakdown_ent{int(ENT_TARGET*100):03d}_bar.jpg")


if __name__ == "__main__":
    main()
