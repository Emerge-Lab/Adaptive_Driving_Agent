"""Plot offline-eval per-trial scores and adaptation deltas across training,
averaged over seeds, separately per ablation group.

For each of the 3 groups (A=lowEgoPenalty+EntCurriculum, B=lowEgoPenalty,
C=original -2 penalty) emits two figures (4-panel, one panel per entropy_ub):
  * trial_{0..3}_score vs offline_eval_iter
  * ada_delta_trial_{1,2,3}_minus_0 vs offline_eval_iter

Mean and ±1 std band across 3 seeds.

Caches the wandb history to logs/offline_eval/_history_cache.parquet so reruns
don't re-fetch.

Usage:
    python scripts/adaptive/plot_offline_eval.py             # uses cache if present
    python scripts/adaptive/plot_offline_eval.py --refresh   # re-fetch from wandb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "logs" / "offline_eval" / "plots"
CACHE_PATH = REPO_ROOT / "logs" / "offline_eval" / "_history_cache.csv"

WANDB_ENTITY = "emerge_"
WANDB_PROJECT = "adaptive_aligned_v2"

# (group, entropy_ub, seed) -> wid
SWEEP = {
    # Group A: lowEgoPenalty + EntCurriculum (ego penalty -0.5, curriculum ON)
    # 11 wids; nbbbsmyr (0.10, s43) is missing.
    ("A", 0.05, 42): "diocrfd9",
    ("A", 0.05, 43): "se7ovksg",
    ("A", 0.05, 44): "rx3yj0k7",
    ("A", 0.10, 42): "3s24do45",
    ("A", 0.10, 44): "hke6hyik",
    ("A", 0.20, 42): "l9wv41ct",
    ("A", 0.20, 43): "nbipb5q9",
    ("A", 0.20, 44): "ipdv2oag",
    ("A", 0.50, 42): "6rv8gcrr",
    ("A", 0.50, 43): "icmjygwf",
    ("A", 0.50, 44): "rwg5a65x",
    # Group B: lowEgoPenalty (ego penalty -0.5, curriculum OFF)
    ("B", 0.05, 42): "a37ay3nb",
    ("B", 0.05, 43): "t1bkn7fq",
    ("B", 0.05, 44): "umaskfka",
    ("B", 0.10, 42): "5obko2iy",
    ("B", 0.10, 43): "4cowebjw",
    ("B", 0.10, 44): "s1ro6tzv",
    ("B", 0.20, 42): "rmsghbiu",
    ("B", 0.20, 43): "wplaas1l",
    ("B", 0.20, 44): "h7fajqan",
    ("B", 0.50, 42): "0rkojso4",
    ("B", 0.50, 43): "251vz655",
    ("B", 0.50, 44): "uljixs7j",
    # Group C: per-partner (ego penalty -2, no curriculum)
    ("C", 0.05, 42): "mpyo1ucm",
    ("C", 0.05, 43): "5fhn4zng",
    ("C", 0.05, 44): "38g805cy",
    ("C", 0.10, 42): "4tqk602k",
    ("C", 0.10, 43): "he3wmzo4",
    ("C", 0.10, 44): "0da431f2",
    ("C", 0.20, 42): "o13lmh0q",
    ("C", 0.20, 43): "6nrtfrex",
    ("C", 0.20, 44): "auspoa8z",
    ("C", 0.50, 42): "438s7mb2",
    ("C", 0.50, 43): "huk9yuqd",
    ("C", 0.50, 44): "96f15g3o",
}

GROUP_LABELS = {
    "A": "Group A — ego penalty -0.5, entropy curriculum ON",
    "B": "Group B — ego penalty -0.5, no curriculum",
    "C": "Group C — ego penalty -2, no curriculum (original sweep)",
}

ENTROPY_UBS = [0.05, 0.10, 0.20, 0.50]
TRIAL_KEYS = [f"offline_eval/human_replay_trial_{k}_score" for k in range(4)]
DELTA_KEYS = [f"offline_eval/human_replay_ada_delta_trial_{k}_minus_0" for k in (1, 2, 3)]
SCORE_KEY = "offline_eval/human_replay_score"
COLLISION_KEY = "offline_eval/human_replay_collision_rate"
OFFROAD_KEY = "offline_eval/human_replay_offroad_rate"
STEP_KEY = "offline_eval_iter"
ALL_KEYS = [STEP_KEY, *TRIAL_KEYS, *DELTA_KEYS, SCORE_KEY, COLLISION_KEY, OFFROAD_KEY]


def fetch_history(refresh: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not refresh:
        print(f"[fetch] loading cache: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    import wandb

    api = wandb.Api()
    rows = []
    for (group, ent_ub, seed), wid in SWEEP.items():
        path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wid}"
        print(f"[fetch] {group} ent_ub={ent_ub} seed={seed} wid={wid}", flush=True)
        try:
            run = api.run(path)
        except Exception as e:
            print(f"  ERROR fetching {path}: {e}", file=sys.stderr)
            continue
        # samples=100 is well above the expected ~8 offline_eval writes per
        # wid (but a few wids have local + SLURM duplicate writes pushing the
        # total to ~16; samples=10 was too low and wandb's uniform downsample
        # silently dropped some iters). Dedup happens after.
        hist = run.history(keys=ALL_KEYS, samples=100, pandas=True)
        if hist.empty:
            print(f"  WARN empty history for {wid}", file=sys.stderr)
            continue
        hist = hist.dropna(subset=[STEP_KEY])
        for _, record in hist.iterrows():
            row = {"group": group, "entropy_ub": ent_ub, "seed": seed, "wid": wid}
            for k in ALL_KEYS:
                row[k] = record.get(k)
            rows.append(row)
    df = pd.DataFrame(rows)
    # Dedup: smoke-test runs wrote offline_eval_iter=76 multiple times on
    # rwg5a65x/icmjygwf. Keep the last write per (wid, offline_eval_iter).
    df = df.drop_duplicates(subset=["wid", STEP_KEY], keep="last").reset_index(drop=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"[fetch] cached {len(df)} rows → {CACHE_PATH}")
    return df


def plot_per_trial(df: pd.DataFrame, group: str, out_path: Path) -> None:
    sub = df[df["group"] == group]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, ent_ub in zip(axes, ENTROPY_UBS):
        cell = sub[sub["entropy_ub"] == ent_ub]
        n_seeds = cell["seed"].nunique()
        for k, key in enumerate(TRIAL_KEYS):
            agg = cell.groupby(STEP_KEY)[key].agg(["mean", "std"]).reset_index()
            x = agg[STEP_KEY].to_numpy()
            mu = agg["mean"].to_numpy()
            sd = agg["std"].fillna(0).to_numpy()
            ax.plot(x, mu, label=f"trial {k}", linewidth=2)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        ax.set_title(f"entropy_ub = {ent_ub}  (n_seeds={n_seeds})")
        ax.set_xlabel("training iter")
        ax.set_ylabel("trial_K_score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{GROUP_LABELS[group]}\nper-trial score across training (mean ±1 std over seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_deltas(df: pd.DataFrame, group: str, out_path: Path) -> None:
    sub = df[df["group"] == group]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, ent_ub in zip(axes, ENTROPY_UBS):
        cell = sub[sub["entropy_ub"] == ent_ub]
        n_seeds = cell["seed"].nunique()
        for k, key in zip((1, 2, 3), DELTA_KEYS):
            agg = cell.groupby(STEP_KEY)[key].agg(["mean", "std"]).reset_index()
            x = agg[STEP_KEY].to_numpy()
            mu = agg["mean"].to_numpy()
            sd = agg["std"].fillna(0).to_numpy()
            ax.plot(x, mu, label=f"trial {k} − trial 0", linewidth=2)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"entropy_ub = {ent_ub}  (n_seeds={n_seeds})")
        ax.set_xlabel("training iter")
        ax.set_ylabel("ada_delta_trial_K_minus_0")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{GROUP_LABELS[group]}\nin-context adaptation delta across training (mean ±1 std over seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_score_curve(df: pd.DataFrame, group: str, out_path: Path) -> None:
    """One panel, 4 lines (one per entropy_ub) of `score` vs iter, ±1 std band."""
    sub = df[df["group"] == group]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for ent_ub in ENTROPY_UBS:
        cell = sub[sub["entropy_ub"] == ent_ub]
        agg = cell.groupby(STEP_KEY)[SCORE_KEY].agg(["mean", "std"]).reset_index()
        x = agg[STEP_KEY].to_numpy()
        mu = agg["mean"].to_numpy()
        sd = agg["std"].fillna(0).to_numpy()
        ax.plot(x, mu, label=f"entropy_ub={ent_ub}", linewidth=2)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
    ax.set_xlabel("training iter")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{GROUP_LABELS[group]}\ncomposite score across training (mean ±1 std over seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_safety(df: pd.DataFrame, group: str, out_path: Path) -> None:
    """Two panels: collision_rate and offroad_rate vs iter, 4 lines each."""
    sub = df[df["group"] == group]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, key, ylabel in zip(axes, (COLLISION_KEY, OFFROAD_KEY), ("collision_rate", "offroad_rate")):
        for ent_ub in ENTROPY_UBS:
            cell = sub[sub["entropy_ub"] == ent_ub]
            agg = cell.groupby(STEP_KEY)[key].agg(["mean", "std"]).reset_index()
            x = agg[STEP_KEY].to_numpy()
            mu = agg["mean"].to_numpy()
            sd = agg["std"].fillna(0).to_numpy()
            ax.plot(x, mu, label=f"entropy_ub={ent_ub}", linewidth=2)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        ax.set_xlabel("training iter")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"{GROUP_LABELS[group]}\nsafety metrics across training (mean ±1 std over seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_cross_group_bars(df: pd.DataFrame, metric_key: str, ylabel: str, title: str, out_path: Path) -> None:
    """Grouped bar chart at iter 76 — 3 groups × 4 entropy_ub.

    X: entropy_ub. Bars (3 per cell): groups A, B, C. Y: mean ± std across seeds.
    """
    final = df[df[STEP_KEY] == 76]
    # Aggregate: (group, entropy_ub) -> mean, std
    agg = final.groupby(["group", "entropy_ub"])[metric_key].agg(["mean", "std", "count"]).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    bar_width = 0.25
    group_colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    x_positions = np.arange(len(ENTROPY_UBS))
    for i, group in enumerate(("A", "B", "C")):
        means = []
        stds = []
        for ent_ub in ENTROPY_UBS:
            row = agg[(agg["group"] == group) & (agg["entropy_ub"] == ent_ub)]
            means.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
            stds.append(float(row["std"].fillna(0).iloc[0]) if not row.empty else 0.0)
        offset = (i - 1) * bar_width
        ax.bar(
            x_positions + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=3,
            color=group_colors[group],
            label=GROUP_LABELS[group].split(" — ")[0] + " (" + GROUP_LABELS[group].split(" — ")[1] + ")",
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(e) for e in ENTROPY_UBS])
    ax.set_xlabel("entropy_ub")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} at iter 76 (mean ± std over 3 seeds)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch wandb history (ignore cache)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_history(refresh=args.refresh)
    print(f"[main] {len(df)} rows  |  {df['wid'].nunique()} wids  |  "
          f"iters per wid: {df.groupby('wid').size().median():.0f}")
    if df.empty:
        sys.exit("no history rows; nothing to plot")

    for group in ("A", "B", "C"):
        plot_per_trial(df, group, OUT_DIR / f"per_trial_score_group_{group}.png")
        plot_deltas(df, group, OUT_DIR / f"ada_delta_group_{group}.png")
        plot_score_curve(df, group, OUT_DIR / f"score_group_{group}.png")
        plot_safety(df, group, OUT_DIR / f"safety_group_{group}.png")

    plot_cross_group_bars(
        df,
        metric_key="offline_eval/human_replay_trial_3_score",
        ylabel="trial_3_score",
        title="Cross-group: trial_3_score",
        out_path=OUT_DIR / "cross_group_trial_3_score_at_iter76.png",
    )
    plot_cross_group_bars(
        df,
        metric_key="offline_eval/human_replay_ada_delta_trial_3_minus_0",
        ylabel="ada_delta_trial_3_minus_0",
        title="Cross-group: in-context adaptation delta (trial 3 − trial 0)",
        out_path=OUT_DIR / "cross_group_ada_delta_3_at_iter76.png",
    )


if __name__ == "__main__":
    main()
