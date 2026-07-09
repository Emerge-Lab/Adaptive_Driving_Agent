"""Unit test for per-agent transformer_position in _forward_eval_legacy.

Test 1 (equivalence): with no resets, the (B,)-position path must produce
outputs identical to the scalar-position path for the same input stream.

Test 2 (reset semantics): after resetting agent 0's pointer mid-stream, its
subsequent outputs must be bit-identical to a fresh run that saw only the
post-reset observations. This is the property the memory-ablation control
relies on.
"""
import os
os.environ["PUFFER_TRANSFORMER_LEGACY_EVAL"] = "1"

import torch
import torch.nn as nn
from pufferlib.models import TransformerWrapper

OBS, HID, HOR, B, T = 12, 32, 16, 4, 10
torch.manual_seed(0)


class FakeEnv:
    class single_observation_space:
        shape = (OBS,)


class FakePolicy(nn.Module):
    is_continuous = False

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(OBS, HID)
        self.actor = nn.Linear(HID, 5)
        self.value = nn.Linear(HID, 1)

    def encode_observations(self, obs, state=None):
        return self.enc(obs)

    def decode_actions(self, h):
        return self.actor(h), self.value(h)


def run(obs_seq, positions=None, reset_agent=None, reset_at=None):
    torch.manual_seed(1)
    model = TransformerWrapper(FakeEnv(), FakePolicy(), input_size=HID,
                               hidden_size=HID, num_layers=2, num_heads=4, horizon=HOR)
    model.eval()
    state = {}
    if positions is not None:
        state["transformer_position"] = positions.clone()
    outs = []
    with torch.no_grad():
        for t, ob in enumerate(obs_seq):
            logits, _ = model.forward_eval(ob, state)
            outs.append(logits)
            if reset_agent is not None and t == reset_at:
                state["transformer_position"][reset_agent] = 0
                state["transformer_context"][reset_agent] = 0
    return outs


obs_seq = [torch.randn(B, OBS) for _ in range(T)]

# --- Test 1: scalar vs vector equivalence (no resets) ---
outs_scalar = run(obs_seq)                                        # scalar pos (default)
outs_vector = run(obs_seq, positions=torch.zeros(B, dtype=torch.long))
maxdiff = max((a - b).abs().max().item() for a, b in zip(outs_scalar, outs_vector))
print(f"TEST1 scalar-vs-vector maxdiff = {maxdiff:.2e}  ->  {'PASS' if maxdiff < 1e-6 else 'FAIL'}")

# --- Test 2: reset agent == fresh episode ---
RESET_AT = 4  # reset agent 0 after step 4 -> steps 5..9 should look fresh
outs_reset = run(obs_seq, positions=torch.zeros(B, dtype=torch.long),
                 reset_agent=0, reset_at=RESET_AT)
# fresh run seeing only the post-reset observations of agent 0
fresh_seq = [ob[0:1] for ob in obs_seq[RESET_AT + 1:]]
outs_fresh = run(fresh_seq, positions=torch.zeros(1, dtype=torch.long))
maxdiff2 = max((a[0] - b[0]).abs().max().item()
               for a, b in zip(outs_reset[RESET_AT + 1:], outs_fresh))
print(f"TEST2 reset-vs-fresh maxdiff = {maxdiff2:.2e}  ->  {'PASS' if maxdiff2 < 1e-5 else 'FAIL'}")

# --- Test 3: non-reset agents unaffected by agent 0's reset ---
maxdiff3 = max((a[1:] - b[1:]).abs().max().item()
               for a, b in zip(outs_reset, outs_vector))
print(f"TEST3 other-agents-unaffected maxdiff = {maxdiff3:.2e}  ->  {'PASS' if maxdiff3 < 1e-6 else 'FAIL'}")
