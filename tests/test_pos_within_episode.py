"""Unit tests for TransformerWrapper.compute_pos_within_episode.

Validates the per-episode position-reset formula used by forward() to
align training-time PE indexing with rollout-time forward_eval PE
indexing under multi-episode-per-row segments.

Convention (matches create_episode_mask):
- terminals[b, t] = 1 means slot t is the LAST slot of an episode.
- Slot t+1 starts the next episode (pos_within_episode = 0).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pufferlib.models import TransformerWrapper


def _expect(label, terminals, expected):
    t = torch.tensor(terminals, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    out = TransformerWrapper.compute_pos_within_episode(t).squeeze(0).tolist()
    assert out == expected, f"FAIL {label}: got {out}, expected {expected}"
    print(f"  ok: {label} → {out}")


def test_no_terminals():
    _expect("all zeros", [0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5])


def test_single_terminal_middle():
    _expect("terminal at slot 2", [0, 0, 1, 0, 0, 0], [0, 1, 2, 0, 1, 2])


def test_single_terminal_end():
    _expect("terminal at last slot", [0, 0, 0, 0, 0, 1], [0, 1, 2, 3, 4, 5])


def test_multi_terminals():
    _expect("terminals at 1 and 4", [0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 2, 0])


def test_back_to_back_terminals():
    _expect("terminals at 2 and 3", [0, 0, 1, 1, 0, 0], [0, 1, 2, 0, 0, 1])


def test_terminal_at_slot_zero():
    # Terminal at slot 0 means slot 0 is a 1-slot episode (episode 0);
    # slots 1..5 are episode 1 starting from pos=0.
    _expect("terminal at slot 0", [1, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 4])


def test_batched():
    """Batched input: each row independent."""
    t = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    out = TransformerWrapper.compute_pos_within_episode(t).tolist()
    expected = [
        [0, 1, 2, 0, 1, 2],
        [0, 1, 0, 1, 2, 0],
        [0, 1, 2, 3, 4, 5],
    ]
    assert out == expected, f"FAIL batched: got {out}, expected {expected}"
    print(f"  ok: batched (3 rows independent)")


def test_clamp_against_long_episode():
    """An episode longer than horizon should produce monotonically growing
    pos but our forward() clamps the gather index. Verify formula itself
    doesn't clamp (clamp lives in caller)."""
    t = torch.zeros(1, 10)
    out = TransformerWrapper.compute_pos_within_episode(t).squeeze(0).tolist()
    assert out == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], f"FAIL clamp: got {out}"
    print(f"  ok: long episode formula = arange (no clamp inside formula)")


def test_consistency_with_create_episode_mask():
    """For terminals = [0,0,1,0,0,0] both formulas should agree:
    - episode_ids[t] = sum of terminals[<t]  → which episode slot t is in
    - pos_within_ep[t] = t - first_slot_of_episode_id[t]
    """
    import torch.nn.functional as F

    terminals = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float32)
    # Replicate create_episode_mask's logic for episode_ids
    episode_ids = F.pad(terminals[:, :-1], (1, 0)).cumsum(dim=1).long()  # (1, 6)

    pos = TransformerWrapper.compute_pos_within_episode(terminals).squeeze(0)
    eids = episode_ids.squeeze(0).tolist()

    # For each episode, the first slot's pos must be 0
    for k in set(eids):
        first_slot = eids.index(k)
        assert pos[first_slot].item() == 0, f"episode {k}: first slot has pos {pos[first_slot].item()}"
    print(f"  ok: consistency with episode_ids ({eids})")


def _run_all():
    test_no_terminals()
    test_single_terminal_middle()
    test_single_terminal_end()
    test_multi_terminals()
    test_back_to_back_terminals()
    test_terminal_at_slot_zero()
    test_batched()
    test_clamp_against_long_episode()
    test_consistency_with_create_episode_mask()
    print("\ntest_pos_within_episode: PASS")


if __name__ == "__main__":
    _run_all()
