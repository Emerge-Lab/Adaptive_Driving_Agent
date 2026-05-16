"""End-to-end system inspection for adaptive_drive under gb=3.

Loads a trained ego checkpoint, builds three different env configurations
(co-player, human-replay, ego-only), steps through each, and saves
exhaustive per-tick logs + KV cache snapshots + attention weights + a
render mp4.

Usage:
    python tests/inspect_system.py \
        --checkpoint experiments/puffer_adaptive_drive_1ljfvs9e/model_puffer_adaptive_drive_000030.pt \
        --info       experiments/puffer_adaptive_drive_1ljfvs9e/info.json \
        --n-steps    200 \
        --out        outputs/inspect

The script does NOT use the multi-worker training stack. It runs a single
Serial env in-process so we can directly read entity-level state from
Drive after every step.

Outputs (one directory per mode):
    outputs/inspect/<mode>/
        per_step.jsonl     # tick-by-tick dump (obs sample, action, reward,
                           # terminal, truncation, removed, slot_t, value,
                           # entropy, env_trial_count proxy)
        kv_cache.npz       # k_norms (n_steps, n_layers, n_heads, horizon)
        garbage_mask.npz   # (n_steps, num_agents, horizon) bool
        attn_layer0.npz    # (n_steps, num_agents, n_heads, horizon)
        env_state.jsonl    # per-tick entity positions, removed flags,
                           # current_goal_reached, all ego agents
        render.mp4         # video of the inspected rollout
        summary.txt        # boundary events table (trial-end, episode-end)

The render is also generated via the standard render_videos pipeline for
each mode, so you get the same overlays you'd see in training renders.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pufferlib.vector
import pufferlib.ocean
import pufferlib.models


# ----------------------------------------------------------------------
# CLI + config
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="path to model_*.pt to load into the ego policy")
    p.add_argument("--info", required=True,
                   help="path to info.json from the same training run")
    p.add_argument("--n-steps", type=int, default=200,
                   help="number of env steps per mode (default 200, ~1 episode under k=2 H=201)")
    p.add_argument("--out", default="outputs/inspect",
                   help="output root directory")
    p.add_argument("--modes", nargs="+",
                   default=["coplayer", "human_replay", "ego_only"],
                   choices=["coplayer", "human_replay", "ego_only"],
                   help="which inspection modes to run")
    p.add_argument("--num-agents", type=int, default=8,
                   help="num_agents for the inspection env (smaller than training)")
    p.add_argument("--num-ego", type=int, default=4,
                   help="num_ego_agents for the inspection env (only used with coplayer)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="device for policy forward (cpu keeps probing simple)")
    p.add_argument("--probe-attention", action="store_true", default=True,
                   help="capture per-step attention weights (memory cost)")
    return p.parse_args()


def load_info(info_path: str) -> dict:
    with open(info_path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Env builders
# ----------------------------------------------------------------------

def make_env_kwargs(info: dict, mode: str, args) -> dict:
    """Build env_kwargs matching the original training config for this mode."""
    env_cfg = copy.deepcopy(info.get("env", {}))

    # Shrink agent count for inspection (full 1024 would be unreadable).
    env_cfg["num_agents"] = args.num_agents
    if mode == "coplayer":
        env_cfg["num_ego_agents"] = args.num_ego
        env_cfg["co_player_enabled"] = True
        env_cfg["human_replay_mode"] = False
        env_cfg["external_co_player_actions"] = False  # inline co-player for inspection
    elif mode == "human_replay":
        env_cfg["co_player_enabled"] = False
        env_cfg["human_replay_mode"] = True
        env_cfg["max_controlled_agents"] = 1
        env_cfg.pop("num_ego_agents", None)
        env_cfg.pop("external_co_player_actions", None)
        env_cfg.pop("co_player_policy", None)
    elif mode == "ego_only":
        env_cfg["co_player_enabled"] = False
        env_cfg["human_replay_mode"] = False
        env_cfg.pop("num_ego_agents", None)
        env_cfg.pop("external_co_player_actions", None)
        env_cfg.pop("co_player_policy", None)
    else:
        raise ValueError(mode)

    # The k_scenarios / scenario_length / goal_behavior come straight from info.
    # Confirm gb=3 is set; if not, force it (the inspection is gb=3-specific).
    env_cfg["goal_behavior"] = 3

    # Make sure render is OFF here (we'll render separately via render_videos)
    env_cfg.pop("render_mode", None)

    return env_cfg


def build_vec(env_kwargs: dict, seed: int):
    creator = pufferlib.ocean.env_creator("puffer_adaptive_drive")
    vec = pufferlib.vector.make(
        creator,
        env_kwargs=env_kwargs,
        backend="Serial",
        num_envs=1,
        seed=seed,
    )
    return vec


# ----------------------------------------------------------------------
# Policy loader
# ----------------------------------------------------------------------

def load_ego_policy(info: dict, checkpoint_path: str, vec, device: str):
    """Construct the ego TransformerWrapper and load checkpoint weights.

    Matches the policy architecture recorded in info.json. Returns the
    policy + initial state ready for forward_eval.
    """
    from pufferlib.ocean.torch import Drive as EgoBase

    # Two separate dim sets (defaults match pufferlib/config/ocean/adaptive.ini):
    #   policy block: Drive base encoder (input_size=128, hidden_size=256)
    #   transformer block: TransformerWrapper (input_size=256, hidden_size=256)
    transformer_cfg = info.get("transformer", {}) or {}
    policy_cfg = info.get("policy", {}) or {}
    policy_input = policy_cfg.get("input_size", 128)
    policy_hidden = policy_cfg.get("hidden_size", 256)
    tf_input = transformer_cfg.get("input_size", 256)
    tf_hidden = transformer_cfg.get("hidden_size", 256)
    num_layers = transformer_cfg.get("num_layers", 2)
    num_heads = transformer_cfg.get("num_heads", 4)
    horizon = transformer_cfg.get(
        "horizon",
        info.get("env", {}).get("k_scenarios", 1) * info.get("env", {}).get("scenario_length", 91),
    )

    driver = vec.driver_env
    base = EgoBase(driver, input_size=policy_input, hidden_size=policy_hidden)
    policy = pufferlib.models.TransformerWrapper(
        env=driver,
        policy=base,
        input_size=tf_input,
        hidden_size=tf_hidden,
        num_layers=num_layers,
        num_heads=num_heads,
        horizon=horizon,
        dropout=0.0,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Some keys may be prefixed with "module." from DDP — strip if present.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    policy = policy.to(device)
    policy.eval()
    return policy, horizon, num_layers, num_heads


# ----------------------------------------------------------------------
# Per-step probe + log
# ----------------------------------------------------------------------

def snapshot_env_state(driver, ego_indices):
    """Pull entity-level state directly from Drive after each step."""
    state = []
    for i, agent_idx in enumerate(ego_indices):
        # entities is a Python wrapper around the C array
        try:
            ent = driver.entities[agent_idx] if hasattr(driver, "entities") else None
        except Exception:
            ent = None
        # We can read removed via the SHM-backed array
        rem = bool(driver.removed[i]) if hasattr(driver, "removed") else None
        tend = bool(driver.trial_ended_this_step[i]) if hasattr(driver, "trial_ended_this_step") else None
        state.append({"ego_local_idx": i, "removed": rem, "trial_ended_this_step": tend})
    return state


def step_and_log(vec, policy, n_steps: int, device, horizon, num_layers, num_heads, probe_attn: bool):
    """Run n_steps of policy-driven rollout and capture everything."""
    obs, _ = vec.reset()
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)

    # Under population_play, vec returns obs for ALL agents (egos + co-players)
    # but the ego policy only controls a subset (vec.driver_env.ego_ids).
    # Subset the obs to ego positions; same for the `removed` slice.
    population_play = bool(getattr(vec, "population_play", False))
    if population_play:
        ego_ids = list(getattr(vec.driver_env, "ego_ids", []))
        ego_idx_t = torch.as_tensor(ego_ids, dtype=torch.long, device=device)
        obs_t = obs_t.index_select(0, ego_idx_t)
    else:
        ego_ids = list(range(obs_t.shape[0]))
        ego_idx_t = torch.as_tensor(ego_ids, dtype=torch.long, device=device)
    B = obs_t.shape[0]

    state = policy.init_eval_state(batch_size=B, device=device, dtype=torch.float32)

    rows = []
    env_state_rows = []
    kv_k_norms = []      # (n_steps, num_layers, num_heads, horizon)
    garbage_masks = []   # (n_steps, B, horizon)
    attn_layer0 = []     # (n_steps, B, num_heads, horizon) — layer 0 only

    driver = vec.driver_env
    # Ego local indices: under coplayer/population_play, ego ids may live
    # somewhere specific; under ego_only it's just 0..B-1. We always probe
    # the FIRST agent (index 0) for the detailed log.
    ego_indices = list(range(B))

    for tick in range(n_steps):
        # Per-step probe attention (heavy — only layer 0 stored to keep size down)
        state["_probe_attention"] = bool(probe_attn)
        state["_attn_weights"] = []
        # Provide `removed` from the env every step so train-mask-equivalent
        # garbage_mask gets populated. The Drive env writes vec.removed via
        # the SHM-backed C path, so we read it directly. Under population_play
        # we subset to ego positions; otherwise the whole array.
        if hasattr(vec, "removed") and vec.removed is not None:
            rem_all = np.asarray(vec.removed)
            rem_subset = rem_all[ego_ids] if population_play else rem_all[:B]
            state["removed"] = torch.as_tensor(rem_subset, device=device, dtype=torch.bool)
        else:
            state["removed"] = torch.zeros(B, dtype=torch.bool, device=device)

        with torch.no_grad():
            logits, value = policy.forward_eval(obs_t, state)
            if isinstance(logits, tuple):
                logits = logits[0]
            # Sample (or argmax) — use argmax for determinism in the inspection.
            action = logits.argmax(dim=-1)

        slot_t = int(state["transformer_position"][0])

        # Snapshot KV-cache k-norm per slot, layer 0 (compact)
        # k_cache layout per layer: (B, num_heads, horizon, head_dim)
        k_norm_per_slot = []
        for li, k_cache in enumerate(state["k_cache"]):
            # Mean over agents and over head_dim → (num_heads, horizon)
            k_norms = k_cache.norm(dim=-1).mean(dim=0)  # (num_heads, horizon)
            k_norm_per_slot.append(k_norms.cpu().numpy())
        kv_k_norms.append(np.stack(k_norm_per_slot, axis=0))  # (num_layers, num_heads, horizon)

        # Snapshot garbage_mask
        gm = state["garbage_mask"].cpu().numpy().astype(bool)  # (B, horizon)
        garbage_masks.append(gm)

        # Snapshot attention weights (layer 0 only)
        if probe_attn and state.get("_attn_weights"):
            w0 = state["_attn_weights"][0]["weights"]  # (B, num_heads, 1, horizon)
            attn_layer0.append(w0.squeeze(-2).cpu().numpy())  # (B, num_heads, horizon)
        else:
            attn_layer0.append(None)

        # Tick env. Under population_play, vec.step expects ONLY the ego
        # actions (shape ego_action_space). Otherwise the full action_space.
        target_space = getattr(vec, "ego_action_space", None) if population_play else vec.action_space
        if target_space is None:
            target_space = vec.action_space
        action_np = action.cpu().numpy().astype(target_space.dtype if hasattr(target_space, "dtype") else np.int64)
        if target_space.shape and len(target_space.shape) == 2 and target_space.shape[-1] == 1 and action_np.ndim == 1:
            action_np = action_np[:, None]
        obs, r, d, t_, info = vec.step(action_np)
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        if population_play:
            obs_t = obs_t.index_select(0, ego_idx_t)

        # Build per-tick log row (FIRST AGENT)
        rows.append({
            "tick": tick,
            "slot_t": slot_t,
            "obs0_first6": obs_t[0, :6].cpu().numpy().tolist(),
            "action0": int(action[0].item()),
            "value0": float(value[0].item()),
            "reward0": float(r[0]),
            "terminal0": bool(d[0]),
            "truncation0": bool(t_[0]),
            "removed0": bool(vec.removed[0]) if hasattr(vec, "removed") else None,
            "trial_ended_this_step0": (
                bool(vec.driver_env.trial_ended_this_step[0])
                if hasattr(vec.driver_env, "trial_ended_this_step") else None
            ),
            "garbage_mask0_sum": int(gm[0].sum()),
            "info_keys": [k for d in info for k in (d.keys() if isinstance(d, dict) else [])][:5],
        })

        # Per-tick env-state for ALL agents (used for full visibility)
        env_state_rows.append({
            "tick": tick,
            "removed_per_agent": [bool(vec.removed[i]) if hasattr(vec, "removed") else None
                                   for i in range(B)],
            "trial_ended_per_agent": [
                bool(vec.driver_env.trial_ended_this_step[i])
                if hasattr(vec.driver_env, "trial_ended_this_step") else None
                for i in range(B)
            ],
            "rewards": r.tolist() if hasattr(r, "tolist") else list(r),
            "terminals": [bool(x) for x in d],
            "truncations": [bool(x) for x in t_],
        })

    return rows, env_state_rows, kv_k_norms, garbage_masks, attn_layer0


# ----------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------

def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_summary(path: Path, rows: list, env_rows: list, mode: str):
    """Boundary events: every step where terminal=1 or trunc=1 fired."""
    boundary_events = []
    for r in rows:
        if r["terminal0"] or r["truncation0"] or r["trial_ended_this_step0"]:
            boundary_events.append(r)
    n_term = sum(1 for r in rows if r["terminal0"])
    n_trunc = sum(1 for r in rows if r["truncation0"])
    n_trial_end = sum(1 for r in rows if r["trial_ended_this_step0"])
    n_limbo_steps = sum(1 for r in rows if r["removed0"])
    total = len(rows)

    cum_reward = sum(r["reward0"] for r in rows)
    n_pos_reward = sum(1 for r in rows if r["reward0"] > 0.01)
    n_neg_reward = sum(1 for r in rows if r["reward0"] < -0.01)

    summary_lines = [
        f"== inspect_system summary: mode={mode} =="
        f"\nticks: {total}",
        f"\nfirst-agent terminals: {n_term}",
        f"\nfirst-agent truncations: {n_trunc}",
        f"\nfirst-agent trial_ended_this_step: {n_trial_end}",
        f"\nfirst-agent steps in limbo (removed=1): {n_limbo_steps}",
        f"\nfirst-agent cumulative reward: {cum_reward:.3f}",
        f"\nfirst-agent positive reward ticks: {n_pos_reward}",
        f"\nfirst-agent negative reward ticks: {n_neg_reward}",
        "",
        "Boundary events (terminal | truncation | trial_ended) for agent 0:",
    ]
    for r in boundary_events:
        summary_lines.append(
            f"  tick={r['tick']:4d}  slot_t={r['slot_t']:3d}  "
            f"term={int(r['terminal0'])}  trun={int(r['truncation0'])}  "
            f"trial_end={int(r['trial_ended_this_step0'])}  "
            f"removed={int(r['removed0'])}  "
            f"r={r['reward0']:+.3f}  v={r['value0']:+.3f}  "
            f"obs[:3]={[f'{x:+.3f}' for x in r['obs0_first6'][:3]]}"
        )
    path.write_text("\n".join(summary_lines) + "\n")


def save_arrays(out_dir: Path, kv_k_norms, garbage_masks, attn_layer0):
    np.savez_compressed(out_dir / "kv_cache.npz", k_norms=np.stack(kv_k_norms))
    np.savez_compressed(out_dir / "garbage_mask.npz", mask=np.stack(garbage_masks))
    attn_valid = [a for a in attn_layer0 if a is not None]
    if attn_valid:
        np.savez_compressed(out_dir / "attn_layer0.npz", attn=np.stack(attn_valid))


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------

def render_one_mode(info: dict, mode: str, out_dir: Path, args):
    """Use the standard render_videos pipeline for an apples-to-apples
    comparison with what training renders look like."""
    from pufferlib.utils import render_videos
    from pufferlib.ocean.drive.rollout import RenderView

    # Build a minimal config that render_videos understands.
    env_kwargs = make_env_kwargs(info, mode, args)
    # Render path uses its own num_agents cap (≤64).
    config = {
        "env": info.get("env_name", "puffer_adaptive_drive"),
        "package": "ocean",
        "data_dir": str(out_dir.parent.parent),  # outputs/inspect/<mode>/.. → outputs/
        "env_config": env_kwargs,
        "use_rnn": False,
        "render_view_modes": [RenderView.FULL_SIM_STATE],
        "eval": info.get("eval", {}) or {},
    }

    # Build the policy fresh for the render env
    creator = pufferlib.ocean.env_creator("puffer_adaptive_drive")
    render_env = pufferlib.vector.make(
        creator, env_kwargs={**env_kwargs, "render_mode": 1, "num_agents": min(env_kwargs.get("num_agents", 8), 64)},
        backend="Serial", num_envs=1, seed=args.seed,
    )
    policy, _, _, _ = load_ego_policy(info, args.checkpoint, render_env, args.device)
    render_env.close()

    # Mock logger
    class _Logger:
        run_id = f"inspect_{mode}"
        wandb = None

    try:
        render_videos(
            config=config,
            policy=policy,
            logger=_Logger(),
            epoch=0,
            global_step=0,
            device=args.device,
            human_replay=(mode == "human_replay"),
        )
        # render_videos saves to {data_dir}/{env}_{run_id}/renders/
        rendered = list((Path(config["data_dir"]) / f"{config['env']}_inspect_{mode}" / "renders").glob("*.mp4"))
        if rendered:
            for src in rendered:
                dst = out_dir / src.name
                src.rename(dst)
                print(f"  saved render: {dst}")
        else:
            print(f"  [warn] no render produced for {mode}")
    except Exception as e:
        print(f"  [warn] render failed for {mode}: {e}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    info = load_info(args.info)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Write a top-level README
    (out_root / "README.md").write_text(f"""# inspect_system output

Checkpoint: `{args.checkpoint}`
Info: `{args.info}`
Modes: {args.modes}
Steps per mode: {args.n_steps}
Num agents (inspection): {args.num_agents}  (ego: {args.num_ego} under coplayer mode)
Seed: {args.seed}

## Files per mode
- `per_step.jsonl`: first-agent tick-by-tick: obs sample, action, reward,
  terminal, truncation, removed, slot_t, value, garbage_mask sum.
- `env_state.jsonl`: per-tick state for ALL agents (removed, trial_ended,
  rewards, terminals, truncations).
- `kv_cache.npz`: k_norms shape (n_steps, num_layers, num_heads, horizon).
- `garbage_mask.npz`: mask shape (n_steps, num_agents, horizon) bool.
- `attn_layer0.npz`: attention weights at layer 0, shape (n_steps, num_agents, num_heads, horizon).
- `summary.txt`: human-readable boundary-events table for agent 0.
- `render.mp4`: video of the rollout with overlays.
""")

    for mode in args.modes:
        print(f"\n{'=' * 64}\n== Mode: {mode}\n{'=' * 64}")
        out_dir = out_root / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build env + policy
        env_kwargs = make_env_kwargs(info, mode, args)
        print(f"  env_kwargs: {json.dumps({k: v for k, v in env_kwargs.items() if not isinstance(v, dict)}, indent=2)}")

        vec = build_vec(env_kwargs, args.seed)
        policy, horizon, num_layers, num_heads = load_ego_policy(info, args.checkpoint, vec, args.device)

        # Step and log
        print(f"  stepping {args.n_steps} ticks...")
        rows, env_rows, kv_k_norms, garbage_masks, attn_layer0 = step_and_log(
            vec, policy, args.n_steps, args.device, horizon, num_layers, num_heads,
            probe_attn=args.probe_attention,
        )

        # Save
        write_jsonl(out_dir / "per_step.jsonl", rows)
        write_jsonl(out_dir / "env_state.jsonl", env_rows)
        save_arrays(out_dir, kv_k_norms, garbage_masks, attn_layer0)
        write_summary(out_dir / "summary.txt", rows, env_rows, mode)
        print(f"  wrote {len(rows)} rows, kv_cache shape {np.stack(kv_k_norms).shape}")

        vec.close()

        # Render (separate env construction since render needs render_mode=1)
        print(f"  rendering...")
        render_one_mode(info, mode, out_dir, args)

    print(f"\nDone. Inspect outputs in: {out_root}")


if __name__ == "__main__":
    main()
