"""Smoke + naming tests for the unified Python render pipeline.

These exercise the C-binding render hooks (`vec_render`, `vec_set_video_suffix`)
and pin the filename contract: Python passes a full basename via
`set_video_suffix` and the C side writes `<basename>.mp4`. If you break that
contract, training videos start landing on collisions or get the `(null)`
prefix again.

The actual rollout test depends on raylib + ffmpeg + xvfb being available, so
it's skipped if they aren't (CI without GPU still imports cleanly).
"""

import os
import shutil
import subprocess
import sys

import pytest


def _have_xvfb():
    return shutil.which("xvfb-run") is not None and shutil.which("ffmpeg") is not None


def _have_maps():
    return os.path.exists("resources/drive/binaries/training/map_001.bin")


@pytest.mark.skipif(not (_have_xvfb() and _have_maps()), reason="needs xvfb + ffmpeg + map binaries")
def test_render_writes_named_mp4(tmp_path):
    """A single render should produce exactly the basename we asked for."""
    import torch

    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.drive.rollout import RenderContext, RenderView

    if "DISPLAY" not in os.environ:
        pytest.skip("no DISPLAY; run under xvfb-run for this test")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        env = Drive(
            num_agents=4,
            num_maps=1,
            map_dir=os.path.join(cwd, "resources/drive/binaries/training"),
            scenario_length=91,
            render_mode=1,
            conditioning={"type": "none"},
        )
        env.reset()
        basename = "pytest_render_smoke_sim_state"
        env.set_video_suffix(basename, env_id=0)
        for _ in range(5):
            env.render(view_mode=int(RenderView.FULL_SIM_STATE), draw_traces=False, env_id=0)
        env.close()

        produced = sorted(os.listdir("."))
        assert f"{basename}.mp4" in produced, f"expected {basename}.mp4, got {produced}"
    finally:
        os.chdir(cwd)


def test_render_context_default_basename_is_safe():
    """Importing rollout module shouldn't blow up and the dataclass should default."""
    from pufferlib.ocean.drive.rollout import RenderContext, RenderView
    ctx = RenderContext(view_mode=RenderView.FULL_SIM_STATE)
    assert ctx.video_basename == "render"
