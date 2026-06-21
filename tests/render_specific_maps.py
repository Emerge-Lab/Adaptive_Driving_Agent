"""Render specific map_ids (matching the eval's use_all_maps indexing) from a
checkpoint, so videos line up with per-map eval ada_delta.

Eval uses use_all_maps=True + 1 SDC per scene, so eval map_id == env index ==
map_<id>.bin. We replicate that: build the env with use_all_maps=True and
num_agents = max(target_ids)+1, then render ONLY the requested env_ids.

Usage:
  xvfb-run -a python tests/render_specific_maps.py \
    --checkpoint experiments/puffer_adaptive_drive_qxw6c0jh/model_puffer_adaptive_drive_000110.pt \
    --info       experiments/puffer_adaptive_drive_qxw6c0jh/info.json \
    --map-ids 169 449 474 496 --n-steps 804 \
    --map-dir resources/drive/binaries/nuplan_hard \
    --out outputs/inspect/qxw6c0jh_topmaps
"""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import sys
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--map-ids", nargs="+", type=int, required=True)
    ap.add_argument("--n-steps", type=int, default=804)
    ap.add_argument("--map-dir", default="resources/drive/binaries/nuplan_hard")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--capture-attention", action="store_true",
                    help="capture per-step layer-0 attention for the target maps "
                         "(eval-faithful via legacy forward) and dump npz for plot_attention.py")
    ap.add_argument("--no-render", action="store_true",
                    help="skip render calls (use when videos already exist and you only want attention)")
    return ap.parse_args()


def make_env_kwargs(info, num_agents, map_dir, render=True):
    env_cfg = copy.deepcopy(info.get("env", {}))
    env_cfg["num_agents"] = num_agents
    env_cfg["map_dir"] = map_dir
    env_cfg["num_maps"] = min(env_cfg.get("num_maps", 540), 540)
    env_cfg["co_player_enabled"] = False
    env_cfg["human_replay_mode"] = True
    env_cfg["max_controlled_agents"] = 1
    env_cfg.pop("num_ego_agents", None)
    env_cfg.pop("external_co_player_actions", None)
    env_cfg.pop("co_player_policy", None)
    env_cfg["goal_behavior"] = 3
    env_cfg["use_all_maps"] = True   # agent i == map_i.bin  (matches eval)
    env_cfg["render_mode"] = 1 if render else 0
    return env_cfg


def load_policy(info, ckpt_path, vec, device):
    from pufferlib.ocean.torch import Drive as EgoBase
    tcfg = info.get("transformer", {}) or {}
    pcfg = info.get("policy", {}) or {}
    horizon = tcfg.get("horizon",
                       info.get("env", {}).get("k_scenarios", 1)
                       * info.get("env", {}).get("scenario_length", 91))
    driver = vec.driver_env
    base = EgoBase(driver, input_size=pcfg.get("input_size", 128),
                   hidden_size=pcfg.get("hidden_size", 256))
    policy = pufferlib.models.TransformerWrapper(
        env=driver, policy=base,
        input_size=tcfg.get("input_size", 256), hidden_size=tcfg.get("hidden_size", 256),
        num_layers=tcfg.get("num_layers", 2), num_heads=tcfg.get("num_heads", 4),
        horizon=horizon, dropout=0.0,
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    policy.load_state_dict(sd, strict=False)
    return policy.to(device).eval()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    info = json.loads(Path(args.info).read_text())

    target_ids = sorted(set(args.map_ids))
    num_agents = max(target_ids) + 1
    print(f"rendering map_ids {target_ids}  (num_agents={num_agents}, use_all_maps)")

    env_kwargs = make_env_kwargs(info, num_agents, args.map_dir, render=not args.no_render)
    creator = pufferlib.ocean.env_creator("puffer_adaptive_drive")
    vec = pufferlib.vector.make(creator, env_kwargs=env_kwargs, backend="Serial", num_envs=1, seed=0)
    driver = vec.driver_env
    map_ids = list(driver.map_ids)
    # Sanity: with use_all_maps, env index i should map to map_i.
    for tid in target_ids:
        if tid < len(map_ids):
            print(f"  env {tid} -> map {map_ids[tid]}")

    # set per-env video suffix only for the targets
    if not args.no_render:
        for tid in target_ids:
            driver.set_video_suffix(f"_map{tid:03d}", env_id=tid)

    policy = load_policy(info, args.checkpoint, vec, args.device)
    obs, _ = vec.reset()
    view = int(RenderView.FULL_SIM_STATE)

    # Attention capture. The probe runs inside the (legacy, eval-faithful)
    # forward_eval and appends one record per layer per step to
    # state["_attn_weights"]; we keep only layer 0 sliced to the target maps,
    # then clear it each step so host RAM stays at O(n_targets) not O(num_agents).
    n_tgt = len(target_ids)
    state = {"_probe_attention": True, "_attn_weights": []} if args.capture_attention else {}
    attn_per_step = [] if args.capture_attention else None     # each: (n_tgt, H, horizon)
    active = np.zeros((args.n_steps, n_tgt), dtype=bool) if args.capture_attention else None

    for t in range(args.n_steps):
        with torch.no_grad():
            ob = torch.as_tensor(obs).to(args.device)
            logits, _ = policy.forward_eval(ob, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(vec.action_space.shape)

        if args.capture_attention:
            rec0 = next(r for r in state["_attn_weights"] if r["layer"] == 0)
            w = rec0["weights"][target_ids].squeeze(-2).numpy()   # (n_tgt, H, horizon)
            attn_per_step.append(w)
            state["_attn_weights"].clear()
            for col, tid in enumerate(target_ids):
                active[t, col] = not bool(driver.removed[tid])

        if not args.no_render:
            for tid in target_ids:
                driver.render(view_mode=view, draw_traces=True, env_id=tid)
        obs, _, _, _, _ = vec.step(action_np)
        if (t + 1) % 100 == 0:
            print(f"  step {t+1}/{args.n_steps}", flush=True)

    if args.capture_attention:
        attn_layer0 = np.stack(attn_per_step, axis=0)             # (T, n_tgt, H, horizon)
        S = attn_layer0.shape[-1]
        gm = np.zeros((args.n_steps, n_tgt, S), dtype=bool)
        for col in range(n_tgt):
            for s in range(min(args.n_steps, S)):
                gm[:, col, s] = not active[s, col]
        np.savez_compressed(args.out / "attn_layer0.npz", attn=attn_layer0)
        np.savez_compressed(args.out / "garbage_mask.npz", mask=gm)
        np.savez_compressed(args.out / "active.npz", active=active)
        last_active = [int(np.where(active[:, c])[0].max()) if active[:, c].any() else -1
                       for c in range(n_tgt)]
        lines = ["col, map_id, last_active"]
        lines += [f"{c:>4} {target_ids[c]:>7} {last_active[c]:>11}" for c in range(n_tgt)]
        (args.out / "summary.txt").write_text("\n".join(lines))
        print(f"saved attn_layer0.npz {attn_layer0.shape} (col->map: "
              f"{ {c: target_ids[c] for c in range(n_tgt)} })", flush=True)

    print("rollout done, closing env (flushing mp4s)...", flush=True)
    vec.close()

    moved = 0
    for c in list(Path(".").glob("*_map*.mp4")):
        shutil.move(str(c), str(args.out / c.name))
        moved += 1
        print(f"  saved: {args.out / c.name}")
    print(f"moved {moved} mp4 files")


if __name__ == "__main__":
    main()
