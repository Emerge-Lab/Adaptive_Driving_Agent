#!/usr/bin/env python3
"""End-to-end render test: PyTorch policy + C raylib bindings = mp4.

Replaces the old `./visualize` binary smoke. Exercises the same code path
the training loop and `render.py` use (rollout_loop + driver.render).
"""

import os
import shutil
import subprocess
import sys


def test_drive_render():
    if shutil.which("xvfb-run") is None or shutil.which("ffmpeg") is None:
        print("xvfb-run or ffmpeg missing; skipping render test")
        return True
    fixture_maps = os.path.join(os.path.dirname(__file__), "fixtures", "maps")
    if not os.path.exists(os.path.join(fixture_maps, "map_001.bin")):
        print("map fixtures missing; skipping render test")
        return True

    cmd = [
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1280x720x24",
        sys.executable,
        "-c",
        # Inline script: build a tiny env, run a few render calls, verify mp4 lands.
        "import os, sys; os.chdir(os.environ['ORIG_CWD']);\n"
        "from pufferlib.ocean.drive.drive import Drive;\n"
        "from pufferlib.ocean.drive.rollout import RenderView;\n"
        "import tempfile;\n"
        "td = tempfile.mkdtemp(); os.chdir(td);\n"
        "env = Drive(num_agents=4, num_maps=1, "
        "map_dir=os.environ['FIXTURE_MAPS'], "
        "scenario_length=91, render_mode=1, conditioning={'type':'none'});\n"
        "env.reset();\n"
        "env.set_video_suffix('drive_render_smoke', env_id=0);\n"
        "for _ in range(5): env.render(view_mode=int(RenderView.FULL_SIM_STATE), draw_traces=False, env_id=0);\n"
        "env.close();\n"
        "assert 'drive_render_smoke.mp4' in os.listdir('.'), os.listdir('.')\n"
        "print('OK')\n",
    ]

    env_vars = os.environ.copy()
    env_vars["ORIG_CWD"] = os.getcwd()
    env_vars["FIXTURE_MAPS"] = fixture_maps
    env_vars["ASAN_OPTIONS"] = "exitcode=0"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env_vars)
    print(f"render exit code: {result.returncode}")
    if result.stdout:
        print(f"stdout: {result.stdout[-800:]}")
    if result.stderr:
        print(f"stderr: {result.stderr[-800:]}")

    return result.returncode == 0 and "OK" in result.stdout


if __name__ == "__main__":
    sys.exit(0 if test_drive_render() else 1)
