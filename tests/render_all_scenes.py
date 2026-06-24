"""Render ALL sub-env scenes AND capture per-step attention in ONE rollout.

Under human_replay with max_controlled_agents=1 and num_agents=N, the C env
holds N internal sub-envs (one per scene). vec_render takes an env_id and
vec_set_video_suffix is per-env_id, so we can write N separate mp4s from a
single rollout. The transformer probe hook captures per-step attention for
all N agents at once — so agent_id ↔ env_id is 1:1 between attention npz
and mp4 filename for direct video↔heatmap comparison.

IMPORTANT: attention/active arrays are saved IMMEDIATELY after the rollout
loop (BEFORE vec.close), so a raylib segfault during cleanup doesn't lose
data.

Outputs (under --out):
    *_env{i}_map*.mp4    one mp4 per scene
    attn_layer0.npz      (T, N, H, horizon) attention weights, layer 0
    garbage_mask.npz     (T, N, horizon) per-agent garbage mask (derived from active)
    active.npz           (T, N) bool: was each agent active at each tick
    summary.txt          per-scene active-step + map_id

Usage:
    xvfb-run -a python tests/render_all_scenes.py \
        --checkpoint experiments/puffer_adaptive_drive_rwg5a65x/model_puffer_adaptive_drive_000076.pt \
        --info       experiments/puffer_adaptive_drive_rwg5a65x/info.json \
        --num-agents 8 --n-steps 804 \
        --out outputs/inspect/rwg5a65x_iter76/all_scenes
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pufferlib.vector
import pufferlib.ocean
import pufferlib.models
import pufferlib.pytorch
from pufferlib.ocean.drive.rollout import RenderView


VIEW_BY_NAME = {
    "sim_state": RenderView.FULL_SIM_STATE,
    "bev": RenderView.BEV_AGENT_OBS,
    "persp": RenderView.AGENT_PERSPECTIVE,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--num-agents", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=804)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--view",
        choices=list(VIEW_BY_NAME),
        default="sim_state",
        help="render view (sim_state = whole map; bev/persp similar size for nuplan)",
    )
    ap.add_argument(
        "--no-render", action="store_true", help="skip render calls, just capture attention (fast: ~2 min vs ~22 min)"
    )
    ap.add_argument(
        "--map-seed", type=int, default=42, help="env map_seed → pins which maps are loaded (reproducible across runs)"
    )
    ap.add_argument(
        "--map-dir", default=None, help="override map_dir from info.json (e.g. resources/drive/binaries/nuplan_hard)"
    )
    return ap.parse_args()


def make_env_kwargs(info: dict, num_agents: int, map_seed: int, render: bool, map_dir: str | None = None) -> dict:
    env_cfg = copy.deepcopy(info.get("env", {}))
    env_cfg["num_agents"] = num_agents
    if map_dir is not None:
        env_cfg["map_dir"] = map_dir
        # nuplan_hard has 540 maps; cap num_maps to what's available so
        # the env doesn't try to load past the directory's end.
        env_cfg["num_maps"] = min(env_cfg.get("num_maps", 540), 540)
    env_cfg["co_player_enabled"] = False
    env_cfg["human_replay_mode"] = True
    env_cfg["max_controlled_agents"] = 1
    env_cfg.pop("num_ego_agents", None)
    env_cfg.pop("external_co_player_actions", None)
    env_cfg.pop("co_player_policy", None)
    env_cfg["goal_behavior"] = 3
    env_cfg["map_seed"] = map_seed  # pin which maps load → reproducible across passes
    env_cfg["render_mode"] = 1 if render else 0  # 0 = no render
    return env_cfg


def load_policy(info: dict, ckpt_path: str, vec, device: str):
    from pufferlib.ocean.torch import Drive as EgoBase

    transformer_cfg = info.get("transformer", {}) or {}
    policy_cfg = info.get("policy", {}) or {}
    horizon = transformer_cfg.get(
        "horizon",
        info.get("env", {}).get("k_scenarios", 1) * info.get("env", {}).get("scenario_length", 91),
    )
    driver = vec.driver_env
    base = EgoBase(driver, input_size=policy_cfg.get("input_size", 128), hidden_size=policy_cfg.get("hidden_size", 256))
    policy = pufferlib.models.TransformerWrapper(
        env=driver,
        policy=base,
        input_size=transformer_cfg.get("input_size", 256),
        hidden_size=transformer_cfg.get("hidden_size", 256),
        num_layers=transformer_cfg.get("num_layers", 2),
        num_heads=transformer_cfg.get("num_heads", 4),
        horizon=horizon,
        dropout=0.0,
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    policy.load_state_dict(sd, strict=False)
    return policy.to(device).eval()


def _print(msg):
    print(msg, flush=True)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    info = json.loads(Path(args.info).read_text())

    env_kwargs = make_env_kwargs(info, args.num_agents, args.map_seed, render=not args.no_render, map_dir=args.map_dir)
    creator = pufferlib.ocean.env_creator("puffer_adaptive_drive")
    vec = pufferlib.vector.make(creator, env_kwargs=env_kwargs, backend="Serial", num_envs=1, seed=args.seed)
    driver = vec.driver_env
    N = driver.num_envs
    map_ids = list(driver.map_ids)
    _print(f"driver.num_envs = {N}")
    _print(f"map_ids = {map_ids}  (pinned via map_seed={args.map_seed})")
    _print(f"view = {args.view}  render_enabled = {not args.no_render}")

    # Per-env video suffix BEFORE first render (only if rendering).
    if not args.no_render:
        for i in range(N):
            suffix = f"_env{i}_map{map_ids[i]:03d}"
            driver.set_video_suffix(suffix, env_id=i)

    policy = load_policy(info, args.checkpoint, vec, args.device)

    obs, _ = vec.reset()
    state = {"_probe_attention": True, "_attn_weights": []}
    active = np.zeros((args.n_steps, args.num_agents), dtype=bool)
    view_mode_int = int(VIEW_BY_NAME[args.view])

    _print("starting rollout...")
    for t in range(args.n_steps):
        with torch.no_grad():
            ob = torch.as_tensor(obs).to(args.device)
            logits, _ = policy.forward_eval(ob, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(vec.action_space.shape)

        if not args.no_render:
            for i in range(N):
                driver.render(view_mode=view_mode_int, draw_traces=True, env_id=i)

        if hasattr(driver, "removed"):
            for a in range(args.num_agents):
                active[t, a] = not bool(driver.removed[a])

        obs, _, _, _, _ = vec.step(action_np)

        if (t + 1) % 100 == 0:
            _print(f"  step {t + 1}/{args.n_steps}")

    _print("rollout done, saving attention BEFORE vec.close (segfault-resistant)")

    # ===== save attention IMMEDIATELY =====
    try:
        n_records = len(state["_attn_weights"])
        num_layers = max(r["layer"] for r in state["_attn_weights"]) + 1
        horizon = state["_attn_weights"][0]["weights"].shape[-1]
        num_heads = state["_attn_weights"][0]["weights"].shape[1]
        T_eff = n_records // num_layers
        attn_layer0 = np.zeros((T_eff, args.num_agents, num_heads, horizon), dtype=np.float32)
        for t in range(T_eff):
            rec = state["_attn_weights"][t * num_layers + 0]
            attn_layer0[t] = rec["weights"].squeeze(-2).cpu().numpy()
        np.savez_compressed(args.out / "attn_layer0.npz", attn=attn_layer0)
        _print(f"  saved attn_layer0.npz shape={attn_layer0.shape}")

        T = args.n_steps
        S = attn_layer0.shape[-1]
        gm = np.zeros((T, args.num_agents, S), dtype=bool)
        for a in range(args.num_agents):
            for s in range(min(T, S)):
                gm[:, a, s] = not active[s, a]
        np.savez_compressed(args.out / "garbage_mask.npz", mask=gm)
        _print(f"  saved garbage_mask.npz shape={gm.shape}")

        np.savez_compressed(args.out / "active.npz", active=active)
        _print(f"  saved active.npz shape={active.shape}")

        # per-scene summary
        last_active = np.full(args.num_agents, -1, dtype=int)
        for a in range(args.num_agents):
            for t in range(args.n_steps):
                if active[t, a]:
                    last_active[a] = t
        lines = ["env_id, map_id, last_active, pct_active"]
        for i in range(N):
            la = int(last_active[i])
            pct = (la + 1) * 100.0 / args.n_steps if la >= 0 else 0
            lines.append(f"{i:>6} {map_ids[i]:>8} {la:>11} {pct:>6.1f}%")
        long_idx = int(np.argmax(last_active))
        lines.append("")
        lines.append(
            f"longest-active: env_id={long_idx}  map={map_ids[long_idx]}  "
            f"last_active={last_active[long_idx]} "
            f"({(last_active[long_idx] + 1) * 100.0 / args.n_steps:.1f}% of {args.n_steps})"
        )
        (args.out / "summary.txt").write_text("\n".join(lines))
        _print("\n=== summary ===")
        _print("\n".join(lines))
    except Exception:
        _print("ERROR saving attention/summary:")
        traceback.print_exc()

    # ===== close env (flushes mp4s); wrap in try since raylib can segfault =====
    _print("\nclosing env (flushing mp4s)...")
    try:
        vec.close()
        _print("  vec.close() ok")
    except Exception:
        _print("  vec.close() raised:")
        traceback.print_exc()

    # ===== move mp4s into args.out =====
    moved = []
    for i in range(N):
        suffix = f"_env{i}_map{map_ids[i]:03d}"
        for m in Path(".").glob(f"*{suffix}*.mp4"):
            dst = args.out / m.name
            try:
                shutil.move(str(m), str(dst))
                moved.append(dst)
            except Exception:
                _print(f"  failed to move {m} -> {dst}")
    _print(f"\nmoved {len(moved)} mp4 files to {args.out}")
    for m in moved:
        _print(f"  {m}")


if __name__ == "__main__":
    main()
