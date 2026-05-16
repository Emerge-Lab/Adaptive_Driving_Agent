"""Render attention + garbage_mask heatmaps from inspect_system outputs.

For each mode in outputs/inspect_v2/{coplayer,human_replay,ego_only}/:
  - attention_over_time.png : (step, source_slot) heatmap of mean-over-heads
    attention weight + mask overlay + trial-boundary markers
  - attention_per_head.png  : per-head attention at 3 chosen ticks
                              (mid-trial-1, mid-trial-2, episode-end)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path("/workspace/ADA/outputs/inspect_v2")
MODES = ["coplayer", "human_replay", "ego_only"]
TRIAL_BOUNDARIES = [200, 401]  # k=2 × per_trial_timeout=201

def load(mode):
    attn = np.load(ROOT / mode / "attn_layer0.npz")["attn"]        # (T, B, H, horizon)
    gm   = np.load(ROOT / mode / "garbage_mask.npz")["mask"]       # (T, B, horizon)
    kn   = np.load(ROOT / mode / "kv_cache.npz")["k_norms"]        # (T, n_layers, H, horizon)
    return attn, gm, kn

def plot_over_time(mode, attn, gm, agent=0):
    """Time-evolution heatmap: y=tick, x=source slot.
    Top:    full mean-over-heads attention (includes the self-attention "ghost diagonal").
    Middle: same attention but with the diagonal hidden — so you can see PAST-attention only.
    Bottom: garbage mask binary heatmap on same axes."""
    T, B, H, S = attn.shape
    a_mean = attn[:, agent].mean(axis=1)  # (T, S)

    # Off-diagonal mean: zero out the (t, t) cell for each row so you only see
    # what the policy attends to in the PAST. The per-step self-spike (head 1)
    # otherwise dominates the visual.
    a_offdiag = a_mean.copy()
    diag_mask = np.eye(min(T, S), dtype=bool)
    if a_offdiag.shape == diag_mask.shape:
        a_offdiag[diag_mask] = 0.0
    else:
        for t in range(min(T, S)):
            a_offdiag[t, t] = 0.0

    g = gm[:, agent].astype(float)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 14), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1]},
    )

    # Panel 1: full attention (with self-spike diagonal)
    im0 = axes[0].imshow(
        a_mean, aspect="auto", origin="upper",
        cmap="viridis", vmin=0, vmax=np.percentile(a_mean[a_mean > 0], 99),
        interpolation="nearest",
    )
    axes[0].set_ylabel("tick (query step)")
    axes[0].set_title(f"{mode}: mean-over-heads attention (agent {agent}) — diagonal = per-step self-spike (head 1)")
    for tb in TRIAL_BOUNDARIES:
        axes[0].axhline(tb, color="white", linestyle="--", linewidth=1, alpha=0.7)
        axes[0].axvline(tb, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    plt.colorbar(im0, ax=axes[0], label="attention weight")

    # Panel 2: off-diagonal — past-only attention
    vmax2 = np.percentile(a_offdiag[a_offdiag > 0], 99) if (a_offdiag > 0).any() else 0.01
    im1 = axes[1].imshow(
        a_offdiag, aspect="auto", origin="upper",
        cmap="viridis", vmin=0, vmax=vmax2,
        interpolation="nearest",
    )
    axes[1].set_ylabel("tick (query step)")
    axes[1].set_title("same, BUT diagonal zeroed → shows PAST-only attention. Dark regions above the diagonal = mask-blocked limbo slots.")
    for tb in TRIAL_BOUNDARIES:
        axes[1].axhline(tb, color="white", linestyle="--", linewidth=1, alpha=0.7)
        axes[1].axvline(tb, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    plt.colorbar(im1, ax=axes[1], label="attention weight")

    # Panel 3: garbage_mask
    im2 = axes[2].imshow(
        g, aspect="auto", origin="upper",
        cmap="Reds", vmin=0, vmax=1, interpolation="nearest",
    )
    axes[2].set_ylabel("tick")
    axes[2].set_xlabel("source slot (cache position)")
    axes[2].set_title("garbage_mask: True (red) = slot was limbo (= these source slots are MASKED at later attention)")
    for tb in TRIAL_BOUNDARIES:
        axes[2].axhline(tb, color="black", linestyle="--", linewidth=1, alpha=0.7)
        axes[2].axvline(tb, color="black", linestyle=":", linewidth=1, alpha=0.7)
    plt.colorbar(im2, ax=axes[2], label="garbage")

    fig.tight_layout()
    out = ROOT / mode / "attention_over_time.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_per_head(mode, attn, gm, agent=0, ticks=(100, 250, 401)):
    """Per-head attention at 3 ticks. Each row = tick, each col = head."""
    T, B, H, S = attn.shape
    fig, axes = plt.subplots(len(ticks), H, figsize=(4*H, 3.5*len(ticks)), squeeze=False)
    for r, t in enumerate(ticks):
        if t >= T:
            continue
        # Pre-current garbage mask (the one ACTUALLY masked at attention time)
        gm_pre = gm[t-1, agent] if t > 0 else np.zeros(S, dtype=bool)
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
            ax.axvline(t_in, color="orange", linestyle="-", linewidth=1.5, alpha=0.8,
                       label=f"slot {t_in} (limbo-entry)")
            # Mark current slot for this step
            ax.axvline(t, color="cyan", linestyle="--", linewidth=1.0, alpha=0.6,
                       label=f"current slot = {t}")
            # Highlight slot t_in's attention weight specifically
            w_at_tin = w[t_in]
            ax.text(t_in + 5, w_at_tin if w_at_tin > 0 else 0.001,
                    f"w@{t_in}={w_at_tin:.4f}", fontsize=8, color="darkred")
            # Limit view: zoom around limbo region for readability
            ax.set_xlim(max(0, t_in - 30), min(S, t + 5))
            ax.set_ylim(0, max(w[max(0, t_in-30):min(S, t+5)].max() * 1.15, 0.02))
            if col == 0:
                ax.set_title(f"head {h}", fontsize=11)
            if h == 0:
                row_label = (
                    f"step t={t} (limbo entry):\n"
                    f"slot {t_in} NOT yet masked"
                ) if col == 0 else (
                    f"step t={t} (one later):\n"
                    f"slot {t_in} IS masked → 0"
                )
                ax.set_ylabel(row_label, fontsize=10)
            if col == 1:
                ax.set_xlabel("source slot")
    fig.suptitle(
        f"{mode}: ghost-diagonal verification — agent {agent}, limbo entry @ tick {t_in}.\n"
        f"Top: at step {t_in}, slot {t_in} has weight (orange line). Bottom: at step {t_in+1}, slot {t_in} weight MUST be ~0.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / mode / "attention_limbo_verify.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out

def main():
    for mode in MODES:
        print(f"\n== {mode} ==")
        attn, gm, kn = load(mode)
        print(f"  attn shape: {attn.shape}  gm: {gm.shape}  kn: {kn.shape}")
        agent = 0
        p1 = plot_over_time(mode, attn, gm, agent=agent)
        print(f"  saved: {p1}")
        p2 = plot_per_head(mode, attn, gm, agent=agent, ticks=(100, 250, 401))
        print(f"  saved: {p2}")
        # Limbo verify: pick agent 0 if it goes limbo, else the first agent that does
        verify_agent = agent
        if find_limbo_entry(gm, verify_agent) < 0:
            for a in range(gm.shape[1]):
                if find_limbo_entry(gm, a) >= 0:
                    verify_agent = a
                    print(f"  agent 0 never goes limbo; using agent {a} for limbo-verify")
                    break
        p3 = plot_limbo_verify(mode, attn, gm, agent=verify_agent)
        if p3:
            print(f"  saved: {p3}  (verifies slot at limbo-entry goes from non-zero → 0 next step)")

if __name__ == "__main__":
    main()
