"""Map-dir propagation tests.

We've been bitten repeatedly by eval silently switching datasets when only
`--env.map-dir` was set. These tests pin the propagation contract:

  1. The Drive env loads from the `map_dir` it was constructed with.
  2. `pufferl.eval()` inherits `env.map_dir` when `eval.map_dir` is unset.
  3. The eval-subprocess command builders forward a concrete `--eval.map-dir`
     so the child process can't fall back to the ini default.
"""

import os

import pytest

from pufferlib.ocean.drive.drive import Drive
from pufferlib import utils as puffer_utils


NUPLAN = "resources/drive/binaries/nuplan"
WOMD = "resources/drive/binaries/training"
FIXTURE_MAPS = os.path.join(os.path.dirname(__file__), "fixtures", "maps")


def _have(path):
    return os.path.exists(os.path.join(path, "map_001.bin"))


def test_env_uses_constructor_map_dir():
    if not _have(FIXTURE_MAPS):
        pytest.skip("fixture maps missing")
    env = Drive(num_agents=4, num_maps=1, map_dir=FIXTURE_MAPS, conditioning={"type": "none"}, scenario_length=91)
    assert env.map_dir == FIXTURE_MAPS
    env.close()


def test_eval_inherits_env_map_dir_when_eval_unset(monkeypatch, tmp_path):
    """eval() should set env.map_dir = eval.map_dir, and when eval.map_dir is unset
    (None / empty / 'None') it should fall back to env.map_dir."""

    # Stub out the heavy parts of pufferl.eval so we only exercise the
    # map-dir resolution prelude.
    from pufferlib import pufferl

    captured = {}

    def fake_load_env(env_name, args):
        captured["env_map_dir"] = args["env"].get("map_dir")
        captured["eval_map_dir"] = args["eval"].get("map_dir")
        raise SystemExit(0)  # short-circuit before policy/eval work

    monkeypatch.setattr(pufferl, "load_env", fake_load_env)

    args = {
        "env": {"map_dir": NUPLAN},
        "eval": {
            "wosac_realism_eval": False,
            "human_replay_eval": True,
            "map_dir": None,  # not set by user
            "num_maps": 5,
            "human_replay_control_mode": "control_vehicles",
            "backend": "PufferEnv",
        },
        "vec": {"backend": "PufferEnv", "num_envs": 1},
        "load_id": None,
        "save_frames": 0,
    }
    with pytest.raises(SystemExit):
        pufferl.eval(args=args, env_name="puffer_drive")

    assert captured["env_map_dir"] == NUPLAN, f"eval did not inherit env.map_dir; got {captured['env_map_dir']}"
    assert captured["eval_map_dir"] == NUPLAN


def test_human_replay_subprocess_forwards_concrete_map_dir(monkeypatch):
    """run_human_replay_eval_in_subprocess must pass a concrete --eval.map-dir
    so the child can't fall back to its own ini default."""

    captured = {}

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeResult()

    monkeypatch.setattr(puffer_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(puffer_utils.glob, "glob", lambda pattern: ["/tmp/model.pt"])
    monkeypatch.setattr(puffer_utils.os.path, "getctime", lambda p: 0)

    config = {
        "env": "puffer_adaptive_drive",
        "data_dir": "/tmp",
        "env_config": {"map_dir": NUPLAN, "k_scenarios": 2, "conditioning": {"type": "all"}},
        "eval": {
            "human_replay_num_agents": 32,
            "human_replay_num_maps": 50,
            "human_replay_num_rollouts": 100,
            "human_replay_control_mode": "control_vehicles",
            "map_dir": None,
            "num_maps": 20,
        },
    }

    class FakeLogger:
        run_id = "abc"
        wandb = None

    puffer_utils.run_human_replay_eval_in_subprocess(config, FakeLogger(), 0)

    cmd = captured.get("cmd", [])
    assert "--eval.map-dir" in cmd, f"map_dir flag missing: {cmd}"
    idx = cmd.index("--eval.map-dir")
    assert cmd[idx + 1] == NUPLAN, f"forwarded map_dir mismatch: {cmd[idx + 1]}"


def test_wosac_subprocess_forwards_concrete_map_dir(monkeypatch):
    captured = {}

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeResult()

    monkeypatch.setattr(puffer_utils.subprocess, "run", fake_run)
    monkeypatch.setattr(puffer_utils.glob, "glob", lambda pattern: ["/tmp/model.pt"])
    monkeypatch.setattr(puffer_utils.os.path, "getctime", lambda p: 0)

    config = {
        "env": "puffer_drive",
        "data_dir": "/tmp",
        "env_config": {"map_dir": NUPLAN},
        "eval": {"map_dir": None},
    }

    class FakeLogger:
        run_id = "abc"
        wandb = None

    puffer_utils.run_wosac_eval_in_subprocess(config, FakeLogger(), 0)

    cmd = captured.get("cmd", [])
    assert "--eval.map-dir" in cmd, f"map_dir flag missing: {cmd}"
    idx = cmd.index("--eval.map-dir")
    assert cmd[idx + 1] == NUPLAN
