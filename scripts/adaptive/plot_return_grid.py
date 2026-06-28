"""Plot per-trial RETURN curves across seeds for every (entropy_ub, k) cell.

Reads outputs/eval540_return/all_cells_R.csv (combine_eval540_return.py).
Outputs three plots in outputs/eval540_return/:
  g_R_per_trial_curves.jpg    4 x 5 small-multiples (entropy × k). Per cell:
                              one line per seed (per-trial mean return over the
                              539 maps) + cross-seed mean. The HEADLINE plot —
                              if seeds agree on the trial-over-trial trajectory
                              that's real adaptation; if they diverge it's noise.
  g_R_mean_delta_heatmap.jpg  G1-analog: mean per-map ΔR = R_last − R_0 per cell.
  g_R_seed_consistency.jpg    G4-analog: cross-seed Pearson r of per-map ΔR per
                              cell. The reproducibility headline.
"""
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = Path("outputs/eval540_return/all_cells_R.csv")
SUCCESS_CSV = Path("outputs/eval540_combined/all_cells.csv")  # for p_0 filter
OUT_DIR = Path("outputs/eval540_return")
RETURN_DIR = Path("outputs/eval540_return")  # for per-component per-cell CSVs
MANIFEST = Path("scripts/adaptive/final_runs_manifest.csv")
SOLO_ENTROPY = 0.10  # wids not in manifest are the 2e029h15 partner = 0.10 entropy

ENTROPIES = [0.05, 0.10, 0.20, 0.50]
KS = [2, 3, 4, 5, 6]
SEEDS = [42, 43, 44]


def load():
    """data[ent][k][seed] = {map_id: (t0, t1, ..., t_{k-1})}"""
    data = defaultdict(lambda: defaultdict(dict))
    with open(IN_CSV) as f:
        for r in csv.DictReader(f):
            ent = float(r["entropy_ub"])
            k = int(r["k"])
            seed = int(r["seed"])
            mid = int(r["map_id"])
            ts = tuple(float(r[f"t{i}"]) for i in range(k))
            if seed not in data[ent][k]:
                data[ent][k][seed] = {}
            data[ent][k][seed][mid] = ts
    return data


def load_metric_csvs(metric_prefix):
    """Load per_map_{metric_prefix}_k*_seed*_*.csv files (e.g. 'goal', 'crash', 'lane').
    Same wid→entropy join as combine_eval540_return.py. Returns data[ent][k][seed]={mid:tuple},
    or None if no files match (so callers can soft-skip when components weren't logged).
    """
    import glob as _glob
    import re as _re
    pattern = _re.compile(
        rf"per_map_{_re.escape(metric_prefix)}_k(\d+)_seed(\d+)_([0-9a-z]+)\.csv$"
    )
    files = sorted(_glob.glob(str(RETURN_DIR / f"per_map_{metric_prefix}_k*_seed*.csv")))
    if not files:
        return None
    manifest = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            manifest[r["wandb_id"]] = float(r["entropy_ub"])
    data = defaultdict(lambda: defaultdict(dict))
    for f in files:
        m = pattern.search(Path(f).name)
        if m is None:
            continue
        k, seed, wid = int(m.group(1)), int(m.group(2)), m.group(3)
        ent = manifest.get(wid, SOLO_ENTROPY)
        with open(f) as fh:
            for r in csv.DictReader(fh):
                mid = int(float(r["map_id"]))
                ts = tuple(float(r[f"t{i}"]) for i in range(k))
                if seed not in data[ent][k]:
                    data[ent][k][seed] = {}
                data[ent][k][seed][mid] = ts
    return data


def plot_decomp_components_heatmap(data_goal, data_crash, data_lane, keep, fname):
    """3-panel filtered heatmap: per-cell mean ΔX for X ∈ {goal, crash, lane}.
    Replaces the post-hoc Path 1 decomp with the real per-step accumulation from re-eval.
    """
    deltas = [
        (per_cell_delta(data_goal),  "ΔR_goal (+1.0/goal)",
         "Goal-reach contribution to ΔR"),
        (per_cell_delta(data_crash), "ΔR_crash (−0.5/event)",
         "Crash-reduction contribution to ΔR\n(collision + offroad merged)"),
        (per_cell_delta(data_lane),  "ΔR_lane (+0.05/in-lane step)",
         "Lane-time contribution to ΔR"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for ax_idx, (delta_c, cbar_label, title) in enumerate(deltas):
        ax = axes[ax_idx]
        M = np.full((len(ENTROPIES), len(KS)), np.nan)
        for i, ent in enumerate(ENTROPIES):
            for j, k in enumerate(KS):
                res = _filtered_common_delta(delta_c, ent, k, keep)
                if res is None:
                    continue
                _, stack, _ = res
                M[i, j] = float(stack.mean(axis=0).mean())
        vmax = max(0.05, float(np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
        ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
        ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
        for i in range(len(ENTROPIES)):
            for j in range(len(KS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.03f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_title(title, fontsize=11)
    fig.suptitle("ΔR component decomposition on adaptable maps (p_0 < 0.8) — "
                 "from per-step accumulation in re-eval (job 11711843)",
                 fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def print_decomp_components_stats(data_goal, data_crash, data_lane, delta_data, keep):
    dg = per_cell_delta(data_goal)
    dc = per_cell_delta(data_crash)
    dl = per_cell_delta(data_lane)
    print("\n=== ΔR component decomposition (per-step) on adaptable maps (p_0 < 0.8) ===")
    print(f"{'ent':>5} {'k':>2} {'n':>4} {'ΔR':>8} {'goal':>7} {'crash':>7} "
          f"{'lane':>7} {'sum_check':>10}")
    for ent in ENTROPIES:
        for k in KS:
            res = _filtered_common_delta(delta_data, ent, k, keep)
            rg = _filtered_common_delta(dg, ent, k, keep)
            rc = _filtered_common_delta(dc, ent, k, keep)
            rl = _filtered_common_delta(dl, ent, k, keep)
            if any(x is None for x in (res, rg, rc, rl)):
                continue
            n = res[1].shape[1]
            dR = float(res[1].mean(axis=0).mean())
            xg = float(rg[1].mean(axis=0).mean())
            xc = float(rc[1].mean(axis=0).mean())
            xl = float(rl[1].mean(axis=0).mean())
            print(f"{ent:>5.2f} {k:>2} {n:>4} {dR:>+8.3f} {xg:>+7.3f} "
                  f"{xc:>+7.3f} {xl:>+7.3f} {xg+xc+xl:>+10.3f}")


def load_success_full():
    """data_p[ent][k][seed] = {map_id: (p_0, p_1, ..., p_{k-1})}.

    Same source as load_success_p0 but keeps all per-trial success rates.
    """
    data = defaultdict(lambda: defaultdict(dict))
    with open(SUCCESS_CSV) as f:
        for r in csv.DictReader(f):
            ent = float(r["entropy_ub"])
            k = int(r["k"])
            seed = int(r["seed"])
            mid = int(r["map_id"])
            ts = tuple(float(r[f"t{i}"]) for i in range(k))
            if seed not in data[ent][k]:
                data[ent][k][seed] = {}
            data[ent][k][seed][mid] = ts
    return data


def per_cell_delta_p(data_p):
    """delta_p_data[(ent,k)][seed] = (mids_array, delta_p_array). Mirrors per_cell_delta."""
    out = defaultdict(dict)
    for ent in ENTROPIES:
        for k in KS:
            for seed in SEEDS:
                rows = data_p[ent][k].get(seed, {})
                if not rows:
                    continue
                mids = sorted(rows)
                arr = np.array([rows[m] for m in mids])
                dp = arr[:, -1] - arr[:, 0]
                out[(ent, k)][seed] = (np.array(mids), dp)
    return out


def plot_decomp_heatmaps(delta_data, delta_p_data, keep, fname):
    """Two-panel heatmap on filtered maps:
       ΔR_goal     = Δp × 1.0          (gain attributable to more goal-reaches)
       ΔR_non_goal = ΔR_total − ΔR_goal (gain from lane-time + fewer crashes, mixed)
    """
    Mg = np.full((len(ENTROPIES), len(KS)), np.nan)
    Mn = np.full((len(ENTROPIES), len(KS)), np.nan)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            res = _filtered_common_delta(delta_data, ent, k, keep)
            res_p = _filtered_common_delta(delta_p_data, ent, k, keep)
            if res is None or res_p is None:
                continue
            _, stack_R, _ = res
            _, stack_p, _ = res_p
            mean_R = float(stack_R.mean(axis=0).mean())
            mean_p = float(stack_p.mean(axis=0).mean())
            Mg[i, j] = mean_p
            Mn[i, j] = mean_R - mean_p

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        (axes[0], Mg, "ΔR_goal (= Δp × 1.0)",
         "Goal-reach contribution to ΔR"),
        (axes[1], Mn, "ΔR_non_goal (lane + crash, mixed)",
         "Driving-quality contribution to ΔR"),
    ]
    for ax, M, cbar_label, title in panels:
        vmax = max(0.1, float(np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
        ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
        ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
        for i in range(len(ENTROPIES)):
            for j in range(len(KS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.02f}", ha="center", va="center",
                            fontsize=9,
                            color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_title(title, fontsize=11)
    fig.suptitle("ΔR decomposition on adaptable maps (p_0 < 0.8) — "
                 "goal vs non-goal (lane + crash combined)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return Mg, Mn


def print_decomp_stats(delta_data, delta_p_data, keep):
    print("\n=== ΔR decomposition on adaptable maps (p_0 < 0.8) ===")
    print(f"{'ent':>5} {'k':>2} {'n':>4} {'ΔR':>8} {'ΔR_goal':>9} "
          f"{'ΔR_other':>10} {'goal_frac':>10}")
    for ent in ENTROPIES:
        for k in KS:
            res = _filtered_common_delta(delta_data, ent, k, keep)
            res_p = _filtered_common_delta(delta_p_data, ent, k, keep)
            if res is None or res_p is None:
                continue
            _, stack_R, _ = res
            _, stack_p, _ = res_p
            n = stack_R.shape[1]
            dR = float(stack_R.mean(axis=0).mean())
            dG = float(stack_p.mean(axis=0).mean())
            dN = dR - dG
            pct = (dG / dR) if abs(dR) > 1e-6 else float('nan')
            print(f"{ent:>5.2f} {k:>2} {n:>4} {dR:>+8.3f} {dG:>+9.3f} "
                  f"{dN:>+10.3f} {pct:>+9.2f}")


def load_success_p0():
    """p0_data[(ent, k)][seed] = {map_id: p_0 (success rate at trial 0)}.

    Reads outputs/eval540_combined/all_cells.csv (the OLD eval success table —
    same checkpoints, identical to what new eval would emit since reward weights
    don't affect rollouts). Used purely as a SELECTOR for 'room to adapt' so
    selection (on p_0) is statistically independent from measurement (on R).
    """
    data = defaultdict(lambda: defaultdict(dict))
    with open(SUCCESS_CSV) as f:
        for r in csv.DictReader(f):
            ent = float(r["entropy_ub"])
            k = int(r["k"])
            seed = int(r["seed"])
            mid = int(r["map_id"])
            data[(ent, k)][seed][mid] = float(r["t0"])
    return data


def room_to_adapt_filter(p0_data, threshold=0.8):
    """Per cell, set of map_ids where the cross-seed MEAN p_0 < threshold."""
    keep = {}
    for cell, by_seed in p0_data.items():
        seeds = list(by_seed.keys())
        common = sorted(set.intersection(*(set(by_seed[s]) for s in seeds)))
        mean_p0 = {m: float(np.mean([by_seed[s][m] for s in seeds])) for m in common}
        keep[cell] = {m for m, p in mean_p0.items() if p < threshold}
    return keep


def plot_per_trial_mean_std_filtered(data, keep, fname, title_suffix):
    """4 x 5 grid like plot_per_trial_mean_std but only over `keep[(ent,k)]` maps.
    Annotates n maps qualifying per cell in each panel title.
    """
    fig, axes = plt.subplots(len(ENTROPIES), len(KS),
                             figsize=(15, 10), sharey=True)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            ax = axes[i, j]
            allowed = keep.get((ent, k), set())
            d = {}
            for seed in SEEDS:
                rows = data[ent][k].get(seed, {})
                if not rows:
                    continue
                sel = [rows[m] for m in rows if m in allowed]
                if not sel:
                    continue
                d[seed] = np.array(sel).mean(axis=0)
            if len(d) >= 2:
                stack = np.stack([d[s] for s in sorted(d)])
                mean = stack.mean(axis=0)
                std = stack.std(axis=0, ddof=1)
                ax.fill_between(range(k), mean - std, mean + std,
                                color="tab:red", alpha=0.22, label="±1 std")
                ax.plot(range(k), mean, marker="o", lw=2.0, color="tab:red",
                        label="cross-seed mean")
            ax.axhline(0, color="0.5", ls=":", lw=0.7)
            ax.set_title(f"ent={ent:.2f}, k={k} (n={len(allowed)})", fontsize=9)
            ax.set_xticks(range(k))
            ax.grid(True, alpha=0.3)
            if i == len(ENTROPIES) - 1:
                ax.set_xlabel("trial index")
            if j == 0:
                ax.set_ylabel(f"ent={ent:.2f}\nmean return")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle(f"Per-trial mean return on adaptable maps — {title_suffix} "
                 f"(cross-seed mean ± 1 std)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def per_cell_curves(data):
    """curves[(ent,k)][seed] = array of length k = per-trial mean return."""
    out = {}
    for ent in ENTROPIES:
        for k in KS:
            d = {}
            for seed in SEEDS:
                rows = data[ent][k].get(seed, {})
                if not rows:
                    continue
                M = np.array(list(rows.values()))  # (n_maps, k)
                d[seed] = M.mean(axis=0)
            out[(ent, k)] = d
    return out


def per_cell_delta(data):
    """delta_data[(ent,k)][seed] = (mids_array, delta_array)."""
    out = defaultdict(dict)
    for ent in ENTROPIES:
        for k in KS:
            for seed in SEEDS:
                rows = data[ent][k].get(seed, {})
                if not rows:
                    continue
                mids = sorted(rows)
                arr = np.array([rows[m] for m in mids])
                delta = arr[:, -1] - arr[:, 0]
                out[(ent, k)][seed] = (np.array(mids), delta)
    return out


def plot_per_trial_mean_std(curves):
    """4 x 5 grid: cross-seed MEAN line with shaded ±1 std band per cell."""
    fig, axes = plt.subplots(len(ENTROPIES), len(KS),
                             figsize=(15, 10), sharey=True)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            ax = axes[i, j]
            d = curves[(ent, k)]
            if len(d) >= 2:
                stack = np.stack([d[s] for s in sorted(d)])
                mean = stack.mean(axis=0)
                std = stack.std(axis=0, ddof=1)
                ax.fill_between(range(k), mean - std, mean + std,
                                color="tab:blue", alpha=0.22, label="±1 std (3 seeds)")
                ax.plot(range(k), mean, marker="o", lw=2.0, color="tab:blue",
                        label="cross-seed mean")
            ax.axhline(0, color="0.5", ls=":", lw=0.7)
            ax.set_title(f"ent={ent:.2f}, k={k}", fontsize=9)
            ax.set_xticks(range(k))
            ax.grid(True, alpha=0.3)
            if i == len(ENTROPIES) - 1:
                ax.set_xlabel("trial index")
            if j == 0:
                ax.set_ylabel(f"ent={ent:.2f}\nmean return")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle("Per-trial mean return by entropy_ub × k "
                 "(cross-seed mean ± 1 std, 3 seeds)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "g_R_per_trial_mean_std.jpg"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_per_trial_curves(curves):
    fig, axes = plt.subplots(len(ENTROPIES), len(KS),
                             figsize=(15, 10), sharey=True)
    seed_colors = {42: "tab:blue", 43: "tab:orange", 44: "tab:green"}
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            ax = axes[i, j]
            d = curves[(ent, k)]
            for seed in SEEDS:
                if seed in d:
                    ax.plot(range(k), d[seed], marker="o", lw=1.5,
                            color=seed_colors[seed], label=f"s{seed}", alpha=0.85)
            if len(d):
                stack = np.stack([d[s] for s in sorted(d)])
                ax.plot(range(k), stack.mean(axis=0),
                        color="black", lw=2.2, alpha=0.55, label="mean")
            ax.set_title(f"ent={ent:.2f}, k={k}", fontsize=9)
            ax.set_xticks(range(k))
            ax.grid(True, alpha=0.3)
            if i == len(ENTROPIES) - 1:
                ax.set_xlabel("trial index")
            if j == 0:
                ax.set_ylabel(f"ent={ent:.2f}\nmean return")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle("Per-trial mean return by co-player entropy_ub × k "
                 "(3 seeds, black = cross-seed mean)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "g_R_per_trial_curves.jpg"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _common_delta(delta_data, ent, k):
    """Stack delta arrays on the common-map set across seeds. Returns (mids, stack[S, M])."""
    per_seed = delta_data[(ent, k)]
    seeds_sorted = sorted(per_seed)
    sets = [set(per_seed[s][0].tolist()) for s in seeds_sorted]
    common = sorted(set.intersection(*sets))
    stacks = []
    for s in seeds_sorted:
        mids_s, deltas_s = per_seed[s]
        idx_map = {int(m): ix for ix, m in enumerate(mids_s)}
        stacks.append(np.array([deltas_s[idx_map[m]] for m in common]))
    return np.array(common), np.stack(stacks), seeds_sorted


def plot_mean_delta_heatmap(delta_data):
    M = np.full((len(ENTROPIES), len(KS)), np.nan)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            per_seed = delta_data[(ent, k)]
            if len(per_seed) != len(SEEDS):
                continue
            _, stack, _ = _common_delta(delta_data, ent, k)
            M[i, j] = stack.mean(axis=0).mean()

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    vmax = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.03f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, label="mean per-map ΔR (R_last − R_0)")
    ax.set_title("Mean per-map ΔR by entropy_ub × k (continuous return)")
    fig.tight_layout()
    out = OUT_DIR / "g_R_mean_delta_heatmap.jpg"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return M


def _filtered_common_delta(delta_data, ent, k, keep):
    """Common-map stack of per-seed ΔR restricted to allowed map_ids."""
    res = _common_delta(delta_data, ent, k)
    common_mids, stack, seeds_sorted = res
    allowed = keep.get((ent, k), set())
    mask = np.array([int(m) in allowed for m in common_mids])
    if not mask.any():
        return None
    return common_mids[mask], stack[:, mask], seeds_sorted


def plot_mean_delta_heatmap_filtered(delta_data, keep, fname, title):
    M = np.full((len(ENTROPIES), len(KS)), np.nan)
    Ns = np.full((len(ENTROPIES), len(KS)), 0, dtype=int)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            per_seed = delta_data[(ent, k)]
            if len(per_seed) != len(SEEDS):
                continue
            res = _filtered_common_delta(delta_data, ent, k, keep)
            if res is None:
                continue
            _, stack_f, _ = res
            M[i, j] = float(stack_f.mean(axis=0).mean())
            Ns[i, j] = int(stack_f.shape[1])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    vmax = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.02f}\nn={Ns[i,j]}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, label="mean per-map ΔR (R_last − R_0)")
    ax.set_title(title)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return M


def plot_seed_consistency_filtered(delta_data, keep, fname, title):
    M = np.full((len(ENTROPIES), len(KS)), np.nan)
    Ns = np.full((len(ENTROPIES), len(KS)), 0, dtype=int)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            per_seed = delta_data[(ent, k)]
            if len(per_seed) < 2:
                continue
            res = _filtered_common_delta(delta_data, ent, k, keep)
            if res is None:
                continue
            _, stack_f, seeds_sorted = res
            if stack_f.shape[1] < 3:
                continue
            rs = [np.corrcoef(stack_f[a], stack_f[b])[0, 1]
                  for a, b in combinations(range(len(seeds_sorted)), 2)]
            M[i, j] = float(np.mean(rs))
            Ns[i, j] = int(stack_f.shape[1])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    vmax_pos = max(0.3, float(np.nanmax(M)))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax_pos, aspect="auto")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}\nn={Ns[i,j]}",
                        ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.5 * vmax_pos else "black")
    fig.colorbar(im, ax=ax, label="mean cross-seed Pearson r of per-map ΔR")
    ax.set_title(title)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return M


def plot_per_map_delta_histograms(delta_data, keep, fname):
    """Per-cell histogram of cross-seed mean ΔR over the adaptable-map subset.

    Tells us whether the cell-mean ΔR is broad (Gaussian-ish, median ≈ mean)
    or driven by a few miracle maps (heavy positive tail, mean >> median).
    """
    fig, axes = plt.subplots(len(ENTROPIES), len(KS), figsize=(15, 9), sharex=True)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            ax = axes[i, j]
            res = _filtered_common_delta(delta_data, ent, k, keep)
            if res is None:
                ax.axis("off")
                continue
            _, stack, _ = res
            per_map = stack.mean(axis=0)
            ax.hist(per_map, bins=30, color="tab:red", alpha=0.7)
            ax.axvline(0, color="0.5", ls=":", lw=0.7)
            ax.axvline(per_map.mean(), color="black", lw=1.5,
                       label=f"μ={per_map.mean():+.2f}")
            ax.axvline(float(np.median(per_map)), color="tab:green", lw=1.5, ls="--",
                       label=f"med={float(np.median(per_map)):+.2f}")
            ax.set_title(f"ent={ent:.2f}, k={k} (n={len(per_map)})", fontsize=9)
            if i == len(ENTROPIES) - 1:
                ax.set_xlabel("per-map ΔR (cross-seed mean)")
            if j == 0:
                ax.set_ylabel(f"ent={ent:.2f}\ncount")
            ax.legend(fontsize=6, loc="upper right", frameon=False)
    fig.suptitle("Per-map ΔR distribution on adaptable maps (p_0 < 0.8)",
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def print_distribution_stats(delta_data, keep):
    """Per-cell distribution stats: mean, median, std, %>0, %>1, p95, p99, top-5 mean."""
    print("\n=== per-map ΔR distribution on adaptable maps (p_0 < 0.8) ===")
    print(f"{'ent':>5} {'k':>2} {'n':>4} {'mean':>8} {'med':>8} {'std':>6} "
          f"{'%>0':>5} {'%>1':>5} {'p95':>7} {'p99':>7} {'top5_mean':>10}")
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            res = _filtered_common_delta(delta_data, ent, k, keep)
            if res is None:
                continue
            _, stack, _ = res
            per_map = stack.mean(axis=0)
            n = len(per_map)
            mean = float(per_map.mean())
            med = float(np.median(per_map))
            std = float(per_map.std(ddof=1))
            pct_pos = 100 * float((per_map > 0).mean())
            pct_gt1 = 100 * float((per_map > 1.0).mean())
            p95 = float(np.percentile(per_map, 95))
            p99 = float(np.percentile(per_map, 99))
            top5 = float(np.sort(per_map)[-5:].mean())
            print(f"{ent:>5.2f} {k:>2} {n:>4} {mean:>+8.3f} {med:>+8.3f} "
                  f"{std:>6.3f} {pct_pos:>5.1f} {pct_gt1:>5.1f} "
                  f"{p95:>+7.3f} {p99:>+7.3f} {top5:>+10.3f}")


def plot_seed_consistency(delta_data):
    M = np.full((len(ENTROPIES), len(KS)), np.nan)
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            per_seed = delta_data[(ent, k)]
            if len(per_seed) < 2:
                continue
            _, stack, seeds_sorted = _common_delta(delta_data, ent, k)
            rs = [np.corrcoef(stack[a], stack[b])[0, 1]
                  for a, b in combinations(range(len(seeds_sorted)), 2)]
            M[i, j] = float(np.mean(rs))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    vmax_pos = max(0.3, float(np.nanmax(M)))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax_pos, aspect="auto")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(ENTROPIES))); ax.set_yticklabels([f"{e:.2f}" for e in ENTROPIES])
    ax.set_xlabel("adaptation trials k"); ax.set_ylabel("co-player entropy_ub")
    for i in range(len(ENTROPIES)):
        for j in range(len(KS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] < 0.5 * vmax_pos else "black")
    fig.colorbar(im, ax=ax, label="mean cross-seed Pearson r of per-map ΔR")
    ax.set_title("Cross-seed reproducibility of per-map ΔR (high = real signal)")
    fig.tight_layout()
    out = OUT_DIR / "g_R_seed_consistency.jpg"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return M


def summary(g1M, g4M):
    print("\n=== entropy x k summary (continuous return) ===")
    print(f"{'ent':>5} {'k':>2} {'meanΔR':>10} {'xseed_r':>8}")
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            if np.isnan(g1M[i, j]):
                continue
            print(f"{ent:>5.2f} {k:>2} {g1M[i,j]:>+10.04f} {g4M[i,j]:>8.03f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    curves = per_cell_curves(data)
    delta_data = per_cell_delta(data)
    plot_per_trial_curves(curves)
    plot_per_trial_mean_std(curves)
    # adaptable-maps view (select on success p_0 < 0.8 so selection ≠ measurement)
    p0_data = load_success_p0()
    keep = room_to_adapt_filter(p0_data, threshold=0.8)
    plot_per_trial_mean_std_filtered(
        data, keep, "g_R_per_trial_mean_std_room.jpg", "p_0 < 0.8"
    )
    print("\n=== adaptable-map counts per cell (p_0 < 0.8) ===")
    for ent in ENTROPIES:
        line = f"  ent={ent:.2f}: " + " ".join(
            f"k{k}:n={len(keep.get((ent,k), set()))}" for k in KS
        )
        print(line)
    g1M = plot_mean_delta_heatmap(delta_data)
    g4M = plot_seed_consistency(delta_data)
    # FILTERED heatmaps — paper-grade headlines
    g1M_room = plot_mean_delta_heatmap_filtered(
        delta_data, keep, "g_R_mean_delta_heatmap_room.jpg",
        "Mean per-map ΔR — adaptable maps only (p_0 < 0.8)")
    g4M_room = plot_seed_consistency_filtered(
        delta_data, keep, "g_R_seed_consistency_room.jpg",
        "Cross-seed r of per-map ΔR — adaptable maps only (p_0 < 0.8)")
    plot_per_map_delta_histograms(delta_data, keep,
                                  "g_R_per_map_delta_hist_room.jpg")
    print_distribution_stats(delta_data, keep)
    # ----- ΔR decomposition: goal vs non-goal (free, from p + R) -----
    data_p = load_success_full()
    delta_p_data = per_cell_delta_p(data_p)
    plot_decomp_heatmaps(delta_data, delta_p_data, keep,
                         "g_R_decomp_goal_vs_nongoal_room.jpg")
    print_decomp_stats(delta_data, delta_p_data, keep)
    # ----- per-component decomp (Path 2): only if re-eval has produced per_map_{goal,crash,lane}_*.csv -----
    data_goal = load_metric_csvs("goal")
    data_crash = load_metric_csvs("crash")
    data_lane = load_metric_csvs("lane")
    if all(d is not None for d in (data_goal, data_crash, data_lane)):
        plot_decomp_components_heatmap(data_goal, data_crash, data_lane, keep,
                                       "g_R_decomp_components_room.jpg")
        print_decomp_components_stats(data_goal, data_crash, data_lane,
                                      delta_data, keep)
    else:
        print("[plot] component CSVs (per_map_{goal,crash,lane}_*) not present — "
              "skipping 3-panel decomp (re-eval still in flight?)")
    summary(g1M, g4M)
    print("\n=== entropy x k summary (FILTERED: p_0 < 0.8) ===")
    print(f"{'ent':>5} {'k':>2} {'meanΔR':>10} {'xseed_r':>8}")
    for i, ent in enumerate(ENTROPIES):
        for j, k in enumerate(KS):
            if np.isnan(g1M_room[i, j]):
                continue
            print(f"{ent:>5.2f} {k:>2} {g1M_room[i,j]:>+10.04f} {g4M_room[i,j]:>8.03f}")


if __name__ == "__main__":
    main()
