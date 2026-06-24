"""End-to-end equivalence: real co-player checkpoint, real obs shape.

Loads the actual `experiments/puffer_drive_2e029h15.pt` co-player Transformer
(matching the user's command config), generates B=512 random observations,
and steps both the legacy `_forward_eval_legacy` and the KV-cached
`forward_eval` for a long sequence (covering multiple horizon wraps), then
compares logits + values. Runs in fp32 and bf16, on CPU and (if available)
GPU. Also exercises a per-row reset partway through.

Tolerances are looser for bf16 (numerical accumulation) and GPU (kernel
selection differences vs CPU SDPA), but still tight enough to catch logic
bugs as opposed to noise.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from pufferlib.ocean.drive import binding
from scripts.profile_co_player import build_co_player_policy


def _make_obs(B, num_obs, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((B, num_obs), dtype=np.float32)
    # Last feature of each road object is categorical [0, 7); fix it.
    ego_features = (
        num_obs
        - (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        - binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )
    road_start = ego_features + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
    road_view = obs[:, road_start:].reshape(B, binding.MAX_ROAD_SEGMENT_OBSERVATIONS, binding.ROAD_FEATURES)
    road_view[:, :, -1] = rng.integers(0, 7, size=road_view.shape[:2])
    return obs


def _step_seq_legacy(policy, obs_seq, reset_at=None, reset_rows=None):
    state = {}
    outs = []
    with torch.inference_mode():
        for i, obs in enumerate(obs_seq):
            if reset_at is not None and i == reset_at:
                state["transformer_context"][reset_rows] = 0
            logits, value = policy._forward_eval_legacy(obs, state)
            outs.append(
                (tuple(l.clone() for l in logits) if isinstance(logits, tuple) else (logits.clone(),), value.clone())
            )
    return outs


def _step_seq_cached(policy, obs_seq, reset_at=None, reset_rows=None):
    state = {}
    outs = []
    with torch.inference_mode():
        for i, obs in enumerate(obs_seq):
            if reset_at is not None and i == reset_at:
                policy.reset_eval_state(state, done_indices=reset_rows)
            logits, value = policy.forward_eval(obs, state)
            outs.append(
                (tuple(l.clone() for l in logits) if isinstance(logits, tuple) else (logits.clone(),), value.clone())
            )
    return outs


def _max_diff(a, b):
    return float((a.float() - b.float()).abs().max().item())


def run_one(checkpoint, device, dtype, num_steps, B=128, horizon=91, reset_at=None, reset_rows=None):
    torch.manual_seed(0)
    policy, num_obs = build_co_player_policy(horizon=horizon)
    state_dict = torch.load(checkpoint, map_location="cpu")
    policy.load_state_dict(state_dict, strict=True)
    policy = policy.to(device).to(dtype).eval()

    obs_np = _make_obs(B, num_obs, seed=42)
    # Use a fresh obs each step to keep the model exercised, but seeded for repro.
    obs_seq = []
    rng = np.random.default_rng(123)
    for _ in range(num_steps):
        delta = rng.standard_normal(obs_np.shape, dtype=np.float32) * 0.01
        obs_seq.append(torch.from_numpy(obs_np + delta).to(device).to(dtype))

    legacy = _step_seq_legacy(policy, obs_seq, reset_at, reset_rows)
    cached = _step_seq_cached(policy, obs_seq, reset_at, reset_rows)

    worst_logits = 0.0
    worst_value = 0.0
    worst_step = 0
    for i, ((lL_t, vL), (lC_t, vC)) in enumerate(zip(legacy, cached)):
        for lL, lC in zip(lL_t, lC_t):
            d = _max_diff(lL, lC)
            if d > worst_logits:
                worst_logits = d
                worst_step = i
        d = _max_diff(vL, vC)
        if d > worst_value:
            worst_value = d
    return worst_logits, worst_value, worst_step


def main():
    checkpoint = os.environ.get("CO_PLAYER_CKPT", "experiments/puffer_drive_2e029h15.pt")
    if not os.path.exists(checkpoint):
        print(f"SKIP: checkpoint not found at {checkpoint}")
        return 0

    cases = []
    cases.append(("CPU fp32 / 200 steps / no reset", "cpu", torch.float32, 200, None, None, 5e-5))
    cases.append(("CPU bf16 / 200 steps / no reset", "cpu", torch.bfloat16, 200, None, None, 0.1))
    cases.append(("CPU fp32 / 200 steps / reset@95", "cpu", torch.float32, 200, 95, torch.tensor([3, 17, 42]), 5e-5))
    if torch.cuda.is_available():
        cases.append(("CUDA fp32 / 200 steps / no reset", "cuda", torch.float32, 200, None, None, 5e-3))
        cases.append(("CUDA bf16 / 200 steps / no reset", "cuda", torch.bfloat16, 200, None, None, 0.5))

    print(f"Checkpoint: {checkpoint}")
    print(f"{'case':50}  {'worst_logits':>14}  {'worst_value':>14}  {'tol':>10}  {'verdict':>8}")
    failed = 0
    for desc, device, dtype, steps, reset_at, reset_rows, tol in cases:
        try:
            wl, wv, ws = run_one(checkpoint, device, dtype, steps, reset_at=reset_at, reset_rows=reset_rows)
            ok = (wl <= tol) and (wv <= tol)
            print(f"{desc:50}  {wl:>14.3e}  {wv:>14.3e}  {tol:>10.1e}  {'OK' if ok else 'FAIL':>8}")
            if not ok:
                failed += 1
        except Exception as e:
            print(f"{desc:50}  ERROR: {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
