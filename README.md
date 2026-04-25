# Adaptive Driving Agent

A fork of [PufferDrive](https://github.com/Emerge-Lab/PufferDrive) for training
**adaptive** driving policies — agents that infer their partners' behavior
on the fly instead of being told what to expect.

## What we're doing

The training pipeline is two-stage:

1. **Train a population of co-players.** A single conditioned policy
   (`puffer_drive`) is trained over a range of conditioning values
   (entropy weight, discount factor, reward weights). Sampling different
   conditioning vectors during inference gives us a *population* of behaviors
   from one set of weights.
2. **Train an adaptive ego against that population** (`puffer_adaptive_drive`).
   The adaptive policy is *not* conditioned — it has to figure out what kind
   of partner it's facing from the observation stream. Each episode contains
   `k_scenarios` scenarios; the partner is re-sampled at scenario boundaries
   so the agent has multiple shots to adapt within one episode.

Everything runs in PufferLib's RL training stack, with `Drive` (the C+raylib
simulator) for the environment and a PyTorch policy on top.

## Repo layout

```
Adaptive_Driving_Agent/
├── render.py                          # unified Python rendering CLI
├── pufferlib/
│   ├── pufferl.py                     # train/eval entrypoint
│   ├── utils.py                       # render_videos, eval subprocess builders
│   ├── ocean/drive/
│   │   ├── drive.h                    # C simulator + raylib renderer
│   │   ├── drive.py                   # gymnasium wrapper, RenderView, render()
│   │   ├── adaptive.py                # AdaptiveDrivingAgent subclass
│   │   ├── rollout.py                 # arch-agnostic rollout loop (used by render + eval)
│   │   ├── torch.py                   # encoder + LSTMWrapper / TransformerWrapper
│   │   └── binding.{c,h}              # Python C-extension
│   ├── ocean/benchmark/evaluator.py   # HumanReplayEvaluator, WOSACEvaluator
│   └── config/ocean/{drive,adaptive}.ini
├── scripts/
│   ├── coplayers/                     # train co-players (per dataset × arch)
│   ├── adaptive/                      # train adaptive ego using a co-player ckpt
│   ├── baselines/                     # vanilla (no co-player) for comparison
│   └── ablations/
│       ├── all_coplayers.sh           # one-shot driver for the full coplayer matrix
│       └── human_align_ablation.sh    # collision/offroad weight grid
└── tests/                             # pytest suite (see CI section)
```

## Setup

```bash
git clone <this repo>
cd Adaptive_Driving_Agent

uv venv && source .venv/bin/activate
uv pip install -e .

# Build the C extension
python setup.py build_ext --inplace --force

# Headless rendering deps (skip on a desktop with display)
sudo apt install ffmpeg xvfb         # or:
conda install -c conda-forge xorg-x11-server-xvfb-cos6-x86_64 ffmpeg
```

### Map binaries

The simulator reads pre-converted map binaries. Two datasets ship out of
the box:

```
resources/drive/binaries/training/   # WOMD (default)
resources/drive/binaries/nuplan/     # nuPlan
```

To convert your own JSON scenes:

```bash
python -c "from pufferlib.ocean.drive.drive import process_all_maps; \
           process_all_maps('path/to/json/folder', max_maps=5000)"
```

The output lives at `resources/drive/binaries/<folder_name>/map_NNN.bin`.

## Training

The CLI is `puffer train <env_name>`. The env names are:
- `puffer_drive` — co-player / baseline training
- `puffer_adaptive_drive` — adaptive ego against frozen co-player

Architecture is selected with `--policy-architecture {Recurrent, Transformer}`.

### 1. Train the co-player population

Co-players are trained on a single env with conditioning sweeping over the
desired range. The conditioning at inference time is what makes the
population diverse.

Per-dataset slurm scripts (4×4 entropy/discount grid, 16 array tasks each):

```bash
sbatch scripts/coplayers/nuplan_recurrent.sh
sbatch scripts/coplayers/nuplan_transformer.sh
sbatch scripts/coplayers/womd_recurrent.sh
sbatch scripts/coplayers/womd_transformer.sh
```

Or, to fire the full matrix locally (no SLURM):

```bash
# Full budget (2B steps × 8 configs)
bash scripts/ablations/all_coplayers.sh

# Quick smoke (200k steps × 8 configs) — useful to confirm learning happens
bash scripts/ablations/all_coplayers.sh --quick

# Subset
bash scripts/ablations/all_coplayers.sh --datasets nuplan --archs Recurrent
bash scripts/ablations/all_coplayers.sh --cond all
```

Resulting checkpoints land at `experiments/puffer_drive_<wandb_run_id>.pt`.

### 2. Train the adaptive ego

The adaptive scripts wire one co-player checkpoint per array task, then
train the adaptive ego with `co_player_enabled=True`:

```bash
sbatch scripts/adaptive/nuplan_recurrent.sh
sbatch scripts/adaptive/nuplan_transformer.sh
sbatch scripts/adaptive/womd_recurrent.sh
sbatch scripts/adaptive/womd_transformer.sh
```

Before submitting, fill in the `ZIPPED_RUNS=(...)` array in each script with
the co-player checkpoint paths from step 1. Run IDs come from the wandb URLs
of the co-player runs.

The key flags are:
```bash
puffer train puffer_adaptive_drive \
  --policy-architecture Recurrent \           # or Transformer
  --env.k-scenarios 2 \                       # scenarios per episode
  --env.co-player-enabled True \
  --env.co-player-policy.policy-path <ckpt>.pt \
  --env.co-player-policy.architecture Recurrent \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub 0.1 \
  --env.co-player-policy.conditioning.discount-weight-lb 0.8 \
  --env.co-player-policy.conditioning.discount-weight-ub 1.0 \
  --env.map-dir resources/drive/binaries/nuplan \
  --eval.map-dir resources/drive/binaries/nuplan
```

### 3. Baselines (no co-player)

For ablation, train the same architectures with `co_player_enabled=False`:

```bash
sbatch scripts/baselines/nuplan_recurrent.sh
sbatch scripts/baselines/nuplan_transformer.sh
sbatch scripts/baselines/womd_recurrent.sh
sbatch scripts/baselines/womd_transformer.sh
```

Other agents in the scene replay their logged human trajectories.

## Rendering

Single Python entrypoint, `render.py`, replaces the old `./visualize` C
binary and the per-scenario eval scripts. It works for **any** architecture
because policy inference is done in PyTorch (the C side only handles
graphics). Architecture is auto-detected from the checkpoint state-dict.

### Modes

| Flags | What it renders |
|---|---|
| `--model-path X.pt` | Baseline: ego drives, others follow logs |
| `--model-path adaptive.pt --co-player-path coplayer.pt` | Adaptive ego vs frozen co-player |
| `--model-path X.pt --human-replay` | One ego is policy-controlled; everyone else replays human logs |

### Examples

```bash
# Baseline render of a co-player checkpoint
xvfb-run -s "-screen 0 1280x720x24" python render.py \
    --model-path experiments/puffer_drive_<id>.pt \
    --map-dir resources/drive/binaries/training \
    --conditioning-type all \
    --num-renders 5

# Adaptive ego rendered against a co-player population
python render.py \
    --model-path experiments/puffer_adaptive_drive_<id>.pt \
    --co-player-path experiments/puffer_drive_<id>.pt \
    --co-player-conditioning-type all \
    --k-scenarios 2 \
    --num-renders 3 \
    --map-dir resources/drive/binaries/nuplan

# Human-replay: how compatible is the agent with logged humans?
python render.py \
    --model-path experiments/puffer_adaptive_drive_<id>.pt \
    --human-replay \
    --k-scenarios 2 \
    --num-renders 5

# All three views (top-down, BEV, third-person) in one call
python render.py --model-path X.pt --view-mode all
```

### Output

Videos are written to `<run_dir>/renders/`, with descriptive filenames so
multiple modes on the same map don't collide:

```
experiments/puffer_adaptive_drive_<id>/renders/
  <id>_baseline_k1_map002_sim_state.mp4
  <id>_human_replay_k2_map002_sim_state.mp4
  <id>_vs_<coplayer_id>_k2_map002_sim_state.mp4
  <id>_vs_<coplayer_id>_k2_map002_bev.mp4
  <id>_vs_<coplayer_id>_k2_map002_persp.mp4
```

Different renders pick different maps via a prime stride
(`--start-seed`, `--seed-stride`).

### During training

`puffer train ... --train.render True` renders periodically every
`--train.render-interval` epochs through the same path, writes to
`<run_dir>/renders/epoch_<NNN>_<mode>_k<k>_map<id>_<view>.mp4`, and uploads
to wandb.

## Evaluation

Two evaluators ship out of the box:

### Human-replay (compatibility with logged humans)

```bash
puffer eval puffer_drive \
    --eval.human-replay-eval True \
    --load-model-path experiments/puffer_drive_<id>.pt \
    --eval.map-dir resources/drive/binaries/nuplan
```

For adaptive agents, this also reports `ada_delta_*` metrics
(last_scenario − first_scenario) so you can see how much the policy
adapted within an episode.

The training loop runs this automatically every `--eval.eval-interval`
epochs when `--eval.human-replay-eval True`.

### WOSAC (distributional realism)

```bash
puffer eval puffer_drive \
    --eval.wosac-realism-eval True \
    --load-model-path experiments/puffer_drive_<id>.pt
```

### Map-dir handling

`eval.map_dir` defaults to `None` in both `drive.ini` and `adaptive.ini`,
which means **eval inherits the training `env.map_dir`** instead of silently
falling back to a different dataset. To use a different dataset for eval,
set `--eval.map-dir` explicitly.

## Tests + CI

```bash
pip install pytest
pytest tests/test_drive_conditioning.py \
       tests/test_map_dir_flow.py \
       tests/test_render_pipeline.py \
       tests/test_drive_config.py
```

The pytest suite covers:
- **Conditioning shapes** (`tests/test_drive_conditioning.py`):
  every `conditioning_type` × `dynamics_model` combination produces the
  right observation dim and value range.
- **map_dir propagation** (`tests/test_map_dir_flow.py`):
  `eval()` inherits `env.map_dir` when `eval.map_dir` is unset, and the
  human-replay / WOSAC subprocess builders forward a concrete
  `--eval.map-dir` so the child process can't fall back to ini default.
- **Render contract** (`tests/test_render_pipeline.py`):
  `set_video_suffix(name)` produces `name.mp4`. The dataclass default is safe.

GitHub Actions:
- `.github/workflows/install.yml`: pre-commit on every push and PR.
- `.github/workflows/utest.yml`: pytest suite on every push (any branch) and PRs.
- `.github/workflows/render-ci.yml`: end-to-end render smoke (xvfb + ffmpeg).
- `.github/workflows/train-ci.yml`: short training + scenario tests.

Run pre-commit locally before pushing:

```bash
pre-commit install      # one-time
pre-commit run --all-files
```

## Common pitfalls

- **Checkpoint shape mismatch on render.** Conditioning changes the ego
  observation dim, so a checkpoint trained with `conditioning.type=all`
  needs `--conditioning-type all` at render time too.
- **Eval using the wrong dataset.** Either set `--eval.map-dir` explicitly
  or rely on inheritance (the default with `eval.map_dir = None`). Don't set
  `eval.map_dir` to a stale value in your ini.
- **Adaptive batch sizes.** `minibatch_size` must be ≥ `batch_size` and both
  must be divisible by `horizon = k_scenarios * scenario_length` for adaptive.
- **Multiple consecutive renders.** The C extension recreates the raylib
  window per render — that's expected, ~1s overhead per render.
