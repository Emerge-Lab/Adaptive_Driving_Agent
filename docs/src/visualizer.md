# Visualizer

PufferDrive renders headless mp4s from Python. Policy inference runs in PyTorch
and graphics are produced via the C/raylib bindings (`vec_render`), so the same
pipeline works for both LSTM and Transformer policies.

## Dependencies

```bash
sudo apt update
sudo apt install ffmpeg xvfb
```

Without sudo:

```bash
conda install -c conda-forge xorg-x11-server-xvfb-cos6-x86_64 ffmpeg
```

## Run

The unified entrypoint is `render.py` at the repo root:

```bash
# Baseline: ego drives, others follow logged trajectories
xvfb-run -s "-screen 0 1280x720x24" python render.py \
    --model-path experiments/<run>.pt --map-dir resources/drive/binaries/training

# Adaptive ego + frozen co-player population
python render.py --model-path adaptive.pt --co-player-path coplayer.pt \
    --co-player-conditioning-type all --k-scenarios 2

# Human replay: only the SDC is policy-controlled, others = logged
python render.py --model-path X.pt --human-replay --num-renders 5

# Multiple views in one go
python render.py --model-path X.pt --view-mode all
```

Architecture is auto-detected from the checkpoint state-dict; override with
`--policy-architecture {Recurrent,Transformer}`. See `python render.py --help`
for the full set of conditioning, co-player, and rendering flags.
