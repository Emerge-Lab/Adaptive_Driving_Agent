"""Lock-in tests for the conditioning surface.

Conditioning changes the ego observation shape, which has caused checkpoint
load mismatches in the past. These tests pin:
  - the obs dim each conditioning type produces, and
  - the value range of the conditioning slot of the obs vector.

We compute expected dims from the binding constants instead of hard-coded
numbers so a constant bump (e.g. adding lane features) doesn't silently
desync this file.
"""

import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive


BASE_EGO = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}
PARTNER_DIM = (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
ROAD_DIM = binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
FIXTURE_MAPS = os.path.join(os.path.dirname(__file__), "fixtures", "maps")


def expected_obs_dim(dynamics_model: str, conditioning_dims: int) -> int:
    return BASE_EGO[dynamics_model] + conditioning_dims + PARTNER_DIM + ROAD_DIM


def make_env(**kwargs):
    defaults = dict(num_agents=4, num_maps=1, scenario_length=91, map_dir=FIXTURE_MAPS)
    defaults.update(kwargs)
    return Drive(**defaults)


@pytest.mark.parametrize("dynamics_model", ["classic", "jerk"])
def test_no_conditioning(dynamics_model):
    env = make_env(conditioning={"type": "none"}, dynamics_model=dynamics_model)
    assert env.single_observation_space.shape[0] == expected_obs_dim(dynamics_model, 0)
    assert not env.reward_conditioned and not env.entropy_conditioned and not env.discount_conditioned
    obs, _ = env.reset()
    assert obs.shape[1] == expected_obs_dim(dynamics_model, 0)
    env.close()


@pytest.mark.parametrize("dynamics_model", ["classic", "jerk"])
def test_reward_conditioning(dynamics_model):
    env = make_env(
        dynamics_model=dynamics_model,
        conditioning={
            "type": "reward",
            "collision_weight_lb": -1.0,
            "collision_weight_ub": 0.0,
            "offroad_weight_lb": -1.0,
            "offroad_weight_ub": 0.0,
            "goal_weight_lb": 0.0,
            "goal_weight_ub": 1.0,
        },
    )
    base = BASE_EGO[dynamics_model]
    assert env.single_observation_space.shape[0] == expected_obs_dim(dynamics_model, 3)
    assert env.reward_conditioned and not env.entropy_conditioned and not env.discount_conditioned
    obs, _ = env.reset()
    rc = obs[:, base : base + 3]
    assert np.all((rc[:, 0] >= -1.0) & (rc[:, 0] <= 0.0))
    assert np.all((rc[:, 1] >= -1.0) & (rc[:, 1] <= 0.0))
    assert np.all((rc[:, 2] >= 0.0) & (rc[:, 2] <= 1.0))
    env.close()


@pytest.mark.parametrize("dynamics_model", ["classic", "jerk"])
def test_entropy_conditioning(dynamics_model):
    env = make_env(
        dynamics_model=dynamics_model,
        conditioning={"type": "entropy", "entropy_weight_lb": 0.0, "entropy_weight_ub": 0.5},
    )
    base = BASE_EGO[dynamics_model]
    assert env.single_observation_space.shape[0] == expected_obs_dim(dynamics_model, 1)
    assert env.entropy_conditioned and not env.reward_conditioned and not env.discount_conditioned
    obs, _ = env.reset()
    assert np.all((obs[:, base] >= 0.0) & (obs[:, base] <= 0.5))
    env.close()


@pytest.mark.parametrize("dynamics_model", ["classic", "jerk"])
def test_discount_conditioning(dynamics_model):
    env = make_env(
        dynamics_model=dynamics_model,
        conditioning={"type": "discount", "discount_weight_lb": 0.7, "discount_weight_ub": 0.99},
    )
    base = BASE_EGO[dynamics_model]
    assert env.single_observation_space.shape[0] == expected_obs_dim(dynamics_model, 1)
    assert env.discount_conditioned and not env.reward_conditioned and not env.entropy_conditioned
    obs, _ = env.reset()
    assert np.all((obs[:, base] >= 0.7) & (obs[:, base] <= 0.99))
    env.close()


@pytest.mark.parametrize("dynamics_model", ["classic", "jerk"])
def test_all_conditioning(dynamics_model):
    """type='all' adds 5 dims: 3 reward + 1 entropy + 1 discount, in that order."""
    env = make_env(
        dynamics_model=dynamics_model,
        conditioning={
            "type": "all",
            "collision_weight_lb": -1.0,
            "collision_weight_ub": 0.0,
            "offroad_weight_lb": -1.0,
            "offroad_weight_ub": 0.0,
            "goal_weight_lb": 0.0,
            "goal_weight_ub": 1.0,
            "entropy_weight_lb": 0.0,
            "entropy_weight_ub": 0.1,
            "discount_weight_lb": 0.8,
            "discount_weight_ub": 0.99,
        },
    )
    base = BASE_EGO[dynamics_model]
    assert env.single_observation_space.shape[0] == expected_obs_dim(dynamics_model, 5)
    assert env.reward_conditioned and env.entropy_conditioned and env.discount_conditioned
    obs, _ = env.reset()
    rc = obs[:, base : base + 3]
    ec = obs[:, base + 3]
    dc = obs[:, base + 4]
    assert np.all((rc[:, 0] >= -1.0) & (rc[:, 0] <= 0.0))
    assert np.all((rc[:, 1] >= -1.0) & (rc[:, 1] <= 0.0))
    assert np.all((rc[:, 2] >= 0.0) & (rc[:, 2] <= 1.0))
    assert np.all((ec >= 0.0) & (ec <= 0.1))
    assert np.all((dc >= 0.8) & (dc <= 0.99))
    env.close()
