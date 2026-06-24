"""Render attention + garbage_mask heatmaps from inspect_system outputs.

For each mode in <root>/{coplayer,human_replay,ego_only}/:
  - attention_over_time.png : (step, source_slot) heatmap of mean-over-heads
    attention weight + mask overlay + trial-boundary markers
  - attention_per_head.png  : per-head attention at 3 chosen ticks (one per trial)

Usage:
  python tests/plot_attention.py --root outputs/inspect/rwg5a65x_iter76 \
      --k 4 --scen-len 201 --modes human_replay
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dir with mode subdirs (e.g. outputs/inspect/<wid_iter>)")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["coplayer", "human_replay", "ego_only"],
        choices=["coplayer", "human_replay", "ego_only"],
    )
    ap.add_argument("--k", type=int, default=4, help="k_scenarios (= max_trials_per_episode under gb=3)")
    ap.add_argument("--scen-len", type=int, default=201, help="scenario_length (= per_trial_timeout)")
    ap.add_argument("--agent", type=int, default=None, help="which agent to plot (default: auto-pick longest-active)")
    ap.add_argument(
        "--all-agents", action="store_true", help="emit heatmaps for every agent (writes attention_*_env{i}_*.png)"
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="treat --root as a flat dir (no per-mode subdir); used for render_all_scenes outputs",
    )
    return ap.parse_args()


# Filled in main() from CLI args
ROOT: Path = Path(".")
TRIAL_BOUNDARIES: list[int] = []
PER_HEAD_TICKS: tuple[int, ...] = ()


def load(mode, flat=False):
    base = ROOT if flat else ROOT / mode
    attn = np.load(base / "attn_layer0.npz")["attn"]  # (T, B, H, horizon)
    gm = np.load(base / "garbage_mask.npz")["mask"]  # (T, B, horizon)
    # kv_cache.npz is only produced by inspect_system.py, not render_all_scenes.py
    kv_path = base / "kv_cache.npz"
    if kv_path.exists():
        kn = np.load(kv_path)["k_norms"]  # (T, n_layers, H, horizon)
    else:
        kn = None
    active_path = base / "active.npz"
    if active_path.exists():
        active = np.load(active_path)["active"]  # (T, N) bool
    else:
        active = None
    return attn, gm, kn, active


def plot_over_time(mode, attn, gm, agent=0, active=None):
    """Time-evolution heatmap: y=tick, x=source slot.

    Three panels:
      Top:    full mean-over-heads attention (with the per-step self-spike on the diagonal).
      Middle: garbage_mask (red where slot was limbo at the time the row's query ran).
      Bottom: zoomed attention — only the active range, with auto-rescaled colormap.

    Per-agent trial-event lines (computed from the `active` array, NOT the timeout cap):
      * goal-reach (active True→False): bright yellow solid — "trial K ended for this
        agent because it reached the goal".
      * active-restart (active False→True): cyan solid — "env did its trial-end reset
        and the agent is back driving in the next trial".
    Vertical line at the same source slot too (the cache position where this transition
    was recorded).
    """
    T, B, H, S = attn.shape
    a_mean = attn[:, agent].mean(axis=1)  # (T, S)
    g = gm[:, agent].astype(float)  # (T, S) bool→float

    # Per-agent trial events from the `active` array (T, N). Falls back to the
    # diagonal of garbage_mask if `active` is None (older runs).
    if active is not None:
        act = active[:, agent].astype(bool)
    else:
        act = np.array([g[t, t] < 0.5 for t in range(min(T, S))], dtype=bool)
        if len(act) < T:  # pad if needed
            act = np.concatenate([act, np.zeros(T - len(act), dtype=bool)])

    # last-active = last tick where agent was active
    last_active = int(np.where(act)[0].max() + 1) if act.any() else T

    # Transitions
    goal_reach_steps = []  # active True→False (agent went to limbo)
    restart_steps = []  # active False→True (env trial-end reset; agent active again)
    for t in range(1, min(T, len(act))):
        if act[t - 1] and not act[t]:
            goal_reach_steps.append(t)
        elif (not act[t - 1]) and act[t]:
            restart_steps.append(t)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 16),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1, 3]},
    )

    def _trial_lines(ax, max_axis=None, vertical=True):
        """Draw per-agent goal-reach (yellow) + restart (cyan) lines."""
        for tb in goal_reach_steps:
            if max_axis is not None and tb >= max_axis:
                continue
            ax.axhline(tb, color="yellow", linestyle="-", linewidth=1.8, alpha=0.95)
            if vertical:
                ax.axvline(tb, color="yellow", linestyle=":", linewidth=1.2, alpha=0.6)
        for tb in restart_steps:
            if max_axis is not None and tb >= max_axis:
                continue
            ax.axhline(tb, color="cyan", linestyle="-", linewidth=1.8, alpha=0.95)
            if vertical:
                ax.axvline(tb, color="cyan", linestyle=":", linewidth=1.2, alpha=0.6)

    # Panel 1: full attention over the whole rollout
    im0 = axes[0].imshow(
        a_mean,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=np.percentile(a_mean[a_mean > 0], 99),
        interpolation="nearest",
    )
    axes[0].set_ylabel("tick (query step)")
    axes[0].set_xlabel("source slot (cache position)")
    axes[0].set_title(
        f"{mode}: mean-over-heads attention (agent {agent}) — yellow = goal-reach (trial ended for this agent), cyan = trial restart"
    )
    _trial_lines(axes[0])
    plt.colorbar(im0, ax=axes[0], label="attention weight")

    # Panel 2: garbage_mask
    im1 = axes[1].imshow(
        g,
        aspect="auto",
        origin="upper",
        cmap="Reds",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_ylabel("tick")
    axes[1].set_xlabel("source slot (cache position)")
    axes[1].set_title("garbage_mask: True (red) = slot was limbo at this attention time")
    _trial_lines(axes[1])
    plt.colorbar(im1, ax=axes[1], label="garbage")

    # Panel 3: ZOOMED attention — only ticks before this agent first went to limbo,
    # AND only source slots up to that point. Re-scale colormap for that subregion.
    zoom_T = last_active
    zoom_S = last_active
    a_zoom = a_mean[:zoom_T, :zoom_S]
    vmax_zoom = np.percentile(a_zoom[a_zoom > 0], 99) if (a_zoom > 0).any() else 0.01
    im2 = axes[2].imshow(
        a_zoom,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        vmin=0,
        vmax=vmax_zoom,
        interpolation="nearest",
    )
    axes[2].set_ylabel("tick (query step)")
    axes[2].set_xlabel("source slot (cache position)")
    axes[2].set_title(
        f"ZOOMED on active range only (ticks 0..{zoom_T - 1}, slots 0..{zoom_S - 1}) "
        "— colorbar auto-rescaled to this subregion"
    )
    _trial_lines(axes[2], max_axis=zoom_T)
    plt.colorbar(im2, ax=axes[2], label="attention weight")

    fig.tight_layout()
    out = ROOT / mode / "attention_over_time.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_head(mode, attn, gm, agent=0, ticks=(100, 250, 401)):
    """Per-head attention at 3 ticks. Each row = tick, each col = head."""
    T, B, H, S = attn.shape
    fig, axes = plt.subplots(len(ticks), H, figsize=(4 * H, 3.5 * len(ticks)), squeeze=False)
    for r, t in enumerate(ticks):
        if t >= T:
            continue
        # Pre-current garbage mask (the one ACTUALLY masked at attention time)
        gm_pre = gm[t - 1, agent] if t > 0 else np.zeros(S, dtype=bool)
        for h in range(H):
            ax = axes[r, h]
            w = attn[t, agent, h]  # (S,)
            xs = np.arange(S)
            colors = ["red" if gm_pre[s] else "tab:blue" for s in xs]
            ax.bar(xs, w, color=colors, width=1.0, edgecolor="none")
            # Trial boundary markers
            for tb in TRIAL_BOUNDARIES:
                if tb <= t:
                    ax.axvline(tb, color="orange", linestyle=":", linewidth=1)
            # Current slot marker (cyan vertical)
            ax.axvline(t, color="cyan", linestyle="-", linewidth=0.8, alpha=0.5)
            ax.set_title(f"tick={t} head={h}  (current slot={t})", fontsize=9)
            ax.set_xlim(-1, S + 1)
            ax.set_ylim(0, max(w.max() * 1.1, 0.01))
            if h == 0:
                ax.set_ylabel("attention weight")
            if r == len(ticks) - 1:
                ax.set_xlabel("source slot")
    fig.suptitle(f"{mode}: per-head attention (red = pre-current garbage / limbo source)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / mode / "attention_per_head.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def find_limbo_entry(gm, agent):
    """Earliest tick where gm[t, agent, t] transitions False→True (just-entered limbo).
    Returns -1 if agent never goes limbo."""
    T = gm.shape[0]
    for t in range(T):
        # Slot t marked True at the step right after the agent went off-map
        if gm[t, agent, t]:
            return t
    return -1


def plot_limbo_verify(mode, attn, gm, agent):
    """For a chosen limbo-entry tick t, plot per-head attention at t and at t+1
    side-by-side. At step t the "ghost diagonal" (slot t) is unmasked. At step
    t+1 slot t is now garbage_mask=True, so attention to source t MUST be 0.
    """
    T, B, H, S = attn.shape
    t_in = find_limbo_entry(gm, agent)
    if t_in < 0 or t_in >= T - 1:
        print(f"  [skip {mode}: agent {agent} never goes limbo in [0, T-1)]")
        return None

    # Pre-current mask at each step (= gm of previous step)
    def mask_at(step):
        return gm[step - 1, agent] if step > 0 else np.zeros(S, dtype=bool)

    fig, axes = plt.subplots(2, H, figsize=(4 * H, 6.5), sharex=True, sharey=False)
    for col, t in enumerate([t_in, t_in + 1]):
        m = mask_at(t)
        for h in range(H):
            ax = axes[col, h]
            w = attn[t, agent, h]
            xs = np.arange(S)
            colors = ["red" if m[s] else "tab:blue" for s in xs]
            ax.bar(xs, w, color=colors, width=1.0, edgecolor="none")
            # Mark slot t_in (the "ghost diagonal" of step t_in)
            ax.axvline(
                t_in, color="orange", linestyle="-", linewidth=1.5, alpha=0.8, label=f"slot {t_in} (limbo-entry)"
            )
            # Mark current slot for this step
            ax.axvline(t, color="cyan", linestyle="--", linewidth=1.0, alpha=0.6, label=f"current slot = {t}")
            # Highlight slot t_in's attention weight specifically
            w_at_tin = w[t_in]
            ax.text(
                t_in + 5, w_at_tin if w_at_tin > 0 else 0.001, f"w@{t_in}={w_at_tin:.4f}", fontsize=8, color="darkred"
            )
            # Limit view: zoom around limbo region for readability
            ax.set_xlim(max(0, t_in - 30), min(S, t + 5))
            ax.set_ylim(0, max(w[max(0, t_in - 30) : min(S, t + 5)].max() * 1.15, 0.02))
            if col == 0:
                ax.set_title(f"head {h}", fontsize=11)
            if h == 0:
                row_label = (
                    (f"step t={t} (limbo entry):\nslot {t_in} NOT yet masked")
                    if col == 0
                    else (f"step t={t} (one later):\nslot {t_in} IS masked → 0")
                )
                ax.set_ylabel(row_label, fontsize=10)
            if col == 1:
                ax.set_xlabel("source slot")
    fig.suptitle(
        f"{mode}: ghost-diagonal verification — agent {agent}, limbo entry @ tick {t_in}.\n"
        f"Top: at step {t_in}, slot {t_in} has weight (orange line). Bottom: at step {t_in + 1}, slot {t_in} weight MUST be ~0.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / mode / "attention_limbo_verify.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    global ROOT, TRIAL_BOUNDARIES, PER_HEAD_TICKS
    args = parse_args()
    ROOT = args.root
    TRIAL_BOUNDARIES = [args.scen_len * i for i in range(1, args.k)]
    # One tick in the middle of each trial
    PER_HEAD_TICKS = tuple(args.scen_len * i + args.scen_len // 2 for i in range(args.k))
    print(f"root: {ROOT}")
    print(f"trial boundaries: {TRIAL_BOUNDARIES}")
    print(f"per-head ticks: {PER_HEAD_TICKS}")

    import shutil

    for mode in args.modes:
        mode_dir = ROOT if args.flat else ROOT / mode
        if not mode_dir.exists():
            print(f"\n== {mode} ==  SKIP (no dir {mode_dir})")
            continue
        print(f"\n== {mode} ==  dir={mode_dir}")
        attn, gm, kn, active = load(mode, flat=args.flat)
        print(f"  attn shape: {attn.shape}  gm: {gm.shape}")

        T, B, _, _ = attn.shape
        last_active = []
        for a in range(B):
            la = -1
            for t in range(T):
                if not gm[t, a, t]:
                    la = t
            last_active.append(la)
        print(f"  per-agent last_active: {last_active}")

        # Decide which agents to plot
        if args.all_agents:
            agents = list(range(B))
        elif args.agent is None:
            agents = [int(np.argmax(last_active))]
            print(f"  auto-pick: agent {agents[0]} (last_active={last_active[agents[0]]})")
        else:
            agents = [args.agent]

        # The plot functions write to ROOT/{mode}/{stem}.png (fixed name).
        # For --flat, mode is unused as a subdir — we save into ROOT itself.
        # We work around this by temporarily monkey-patching ROOT to mode_dir
        # so plot funcs land their fixed-name pngs there, then rename per-agent.
        save_dir = mode_dir
        # Compute the dir the plot functions WILL use given the current `mode`
        # arg passed: plot_over_time uses `ROOT / mode / "attention_over_time.png"`.
        # If --flat, ROOT itself is mode_dir, and `mode` will be the subdir name.
        # Easiest: just pass a mode value such that ROOT/mode == mode_dir.
        if args.flat:
            # tell plot funcs to use `.` as the mode dir so output lands in ROOT
            plot_mode = "."
        else:
            plot_mode = mode

        for agent in agents:
            print(f"  -- agent {agent} (last_active={last_active[agent]}) --")
            plot_over_time(plot_mode, attn, gm, agent=agent, active=active)
            for stem in ("attention_over_time", "attention_per_head"):
                src = save_dir / f"{stem}.png"
                if src.exists():
                    dst = save_dir / f"{stem}_env{agent}.png"
                    shutil.move(str(src), str(dst))
                    print(f"    saved: {dst}")
            plot_per_head(plot_mode, attn, gm, agent=agent, ticks=PER_HEAD_TICKS)
            for stem in ("attention_per_head", "attention_over_time"):
                src = save_dir / f"{stem}.png"
                if src.exists():
                    dst = save_dir / f"{stem}_env{agent}.png"
                    if not dst.exists():
                        shutil.move(str(src), str(dst))
                        print(f"    saved: {dst}")


if __name__ == "__main__":
    main()
