"""M2 verification: Python-owned trial_ended_this_step buffer is bound by
C and zeroed at the top of c_step.

We don't yet have GOAL_TRIAL behavior in c_step (M3 will add that), so
for now we just verify:
  1. Python allocates the buffer at __init__
  2. C's c_step memsets it to zero on every step (proven by writing
     non-zero values to the buffer between steps and observing that
     they get cleared)
  3. The buffer's pointer survives env_init / vec_reset / vec_step
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_drive_allocates_buffer():
    """Just constructing Drive must produce self.trial_ended_this_step."""
    from pufferlib.ocean.drive import Drive

    env = Drive(
        num_agents=4,
        map_dir="resources/drive/binaries/nuplan_201",
        num_maps=10,
        scenario_length=91,
        ini_file="pufferlib/config/ocean/drive.ini",
    )
    assert hasattr(env, "trial_ended_this_step"), "Drive must expose trial_ended_this_step"
    assert env.trial_ended_this_step.shape == (env.num_agents,), (
        f"shape mismatch: got {env.trial_ended_this_step.shape}, want ({env.num_agents},)"
    )
    assert env.trial_ended_this_step.dtype == bool, "must be bool dtype (1 byte)"
    print(f"  ok: Drive allocated trial_ended_this_step with shape {env.trial_ended_this_step.shape}")
    env.close()


def test_c_zeros_buffer_each_step():
    """Mutate the Python buffer to non-zero, then run a step; C's memset
    in c_step should zero it (since GOAL_TRIAL behavior isn't active —
    no path sets it to 1 yet)."""
    from pufferlib.ocean.drive import Drive

    env = Drive(
        num_agents=4,
        map_dir="resources/drive/binaries/nuplan_201",
        num_maps=10,
        scenario_length=91,
        ini_file="pufferlib/config/ocean/drive.ini",
    )
    env.reset(seed=42)

    # Pollute the buffer Python-side; C should clear on the next step.
    env.trial_ended_this_step[:] = True
    assert env.trial_ended_this_step.all(), "pre-step pollution should hold"

    # Step the env. action shape depends on env config; use zeros which are
    # valid for discrete (idx 0) or float (no-op).
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    env.step(actions)

    # After c_step, memset(env->trial_ended_this_step, 0, ...) must have
    # zeroed the Python buffer (same memory backing).
    assert not env.trial_ended_this_step.any(), (
        f"C did not zero trial_ended_this_step on step "
        f"(values: {env.trial_ended_this_step.tolist()})"
    )
    print(f"  ok: c_step zeroed the buffer after pollution")
    env.close()


def test_buffer_survives_multiple_steps():
    """Sanity: 50 steps, buffer pointer should remain bound, no crashes."""
    from pufferlib.ocean.drive import Drive

    env = Drive(
        num_agents=4,
        map_dir="resources/drive/binaries/nuplan_201",
        num_maps=10,
        scenario_length=91,
        ini_file="pufferlib/config/ocean/drive.ini",
    )
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    for t in range(50):
        env.step(actions)
        # Every step the buffer should be all-zero (no trial-end logic yet)
        assert not env.trial_ended_this_step.any(), f"non-zero at step {t}"
    print(f"  ok: 50 steps, buffer remained zero (no trial-end logic yet, expected)")
    env.close()


def _run_all():
    test_drive_allocates_buffer()
    test_c_zeros_buffer_each_step()
    test_buffer_survives_multiple_steps()
    print("\ntest_trial_ended_buffer: PASS")


if __name__ == "__main__":
    _run_all()
