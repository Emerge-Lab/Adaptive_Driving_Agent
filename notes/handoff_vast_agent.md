# Handoff: Adaptive Driving Agent — satellite agent on vast.ai (4× RTX 5090)

Written 2026-07-09 by the primary (cluster-side) agent. You are a second agent on
a vast.ai box with 4 local RTX 5090s (32 GB VRAM each, direct access, no slurm).
Your job: run the training experiments listed under "Your task menu" without
colliding with the cluster-side work. **Read this whole doc before running anything.**

## 1. Project in one paragraph

We demonstrate AdA-style in-context adaptation (arXiv 2301.07608) in driving:
an ego transformer policy plays k=4 trials per episode on the same nuplan map
(kinematics reset at trial boundaries, transformer K/V memory persists) against
a conditioned co-player, evaluated on 540 interaction-dense human-log maps
(nuplan_hard, 20 rollouts). Headline: per-trial return climbs ~+0.9 to +1.4 on
"adaptable" maps (p0 < 0.8), driven ~85 % by crash reduction. Mechanism
triangulated: own-failure memory (survives with deterministic partner at
e_ub=0.001: ΔR +0.43; a human demonstration in trial 0 adds ~nothing → not
imitation). Curriculum fine-tuning (+1B on hard/frontier maps) = clean negative.
Full state: `notes/paper_analysis.md` — THE living doc, update it when results land.

## 2. Repo / branch / conventions

- Repo: `github.com:Emerge-Lab/Adaptive_Driving_Agent`, branch `mohit/trial-episode-redesign`.
- W&B: entity `emerge_`, project `adaptive_aligned_v2`. Sweep convention: ONE
  shared `--tag` per experiment family; per-run hyperparams go via CLI flags
  (they land in wandb config). Never encode hyperparams in the tag.
- Cluster scripts in `scripts/adaptive/cluster_*.sh` are slurm+singularity; on
  vast.ai strip the sbatch/singularity wrappers and run the inner
  `puffer train ...` command directly (one run per GPU via CUDA_VISIBLE_DEVICES).
- Metric is LOCKED: per-trial return (sum of rewards, training weights:
  lane 0.05, collision/offroad −0.5, goal 1.0). Offline eval:
  `scripts/adaptive/eval_final_540.py` (see `cluster_eval540_curriculum.sh` for
  canonical invocation).

## 3. Setup on the vast box

1. Clone repo, checkout `mohit/trial-episode-redesign`.
2. Python 3.10/3.11 venv. RTX 5090 is Blackwell (sm_120): you need a torch build
   with cu128+ support — verify `torch.cuda.get_device_capability()` works
   before anything else.
3. Build the C env: `NO_TRAIN=1 python setup.py build_ext --inplace`
   (NO_TRAIN=1 skips the CUDA extension, which is only needed for training
   speedups; if you want it, plain `python setup.py build_ext --inplace` with
   `TORCH_CUDA_ARCH_LIST="12.0"`).
4. **Data (not in git — you must transfer):**
   - `resources/drive/binaries/nuplan_201/` (5402 map .bin files, ~a few GB)
   - `resources/drive/binaries/nuplan_hard/` — NOTE: symlink dir on the cluster;
     rsync with `-L` to materialize, or rebuild via
     `scripts/build_nuplan_hard.py --scores scripts/nuplan_201_hardness_scores.csv`
     (scores CSV IS in git).
   - Partner checkpoints: `experiments/puffer_drive_{2e029h15,miku2puk,m2ygolog,6rauydj2}.pt`
   - Rsync from cluster: `mmk9418@<cluster>:/scratch/mmk9418/projects/Adaptive_Driving_Agent/...`
     (ask the user for the host / credentials).
5. Env vars for ALL runs: `PUFFER_TRANSFORMER_LEGACY_EVAL=1` (NEVER flip — see
   pitfalls), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `WANDB_MODE=online`.
6. Rendering needs a virtual display: prefix training with `xvfb-run -a`
   (headless raylib segfaults without it — signal 11 at first render epoch).

## 4. VRAM reality check (5090 = 32 GB, H100 runs used ~67 GB)

The k=4 recipe at NUM_WORKERS=32/NUM_ENVS=32/minibatch 50×804 peaks ~60-67 GB.
On 32 GB you must shrink. Suggested first attempt (validate before committing
to a full run): `--train.max-minibatch-size 20100` (=25×HORIZON, half),
`--vec.num-workers 16 --vec.num-envs 16 --vec.batch-size 16`, keep
`--train.cpu-offload True`. Expect ~half the SPS (≈20-25K); a 3B run ≈ 35-40h.
Confirm with a 10-minute smoke run per GPU (watch `nvidia-smi` peak) before
launching the full sweep. Batch-size changes alter the effective epoch count —
that's fine (batch_size is derived from num_ego×context×workers), but NOTE the
run's config in wandb will differ from cluster cells; keep seeds/tags clean.

## 5. Your task menu (priority order)

**Task A — k=5 standard-recipe rerun (headline-completing, do first).**
The 60-cell grid's k=5 cells ran a "memsafe" variant (different batch geometry)
→ the "k=5 flat" anomaly is confounded. Rerun k=5 on the STANDARD recipe:
partner 2e029h15, ENTROPY_UB=0.10, K_SCENARIOS=5 (HORIZON=1005), 3 seeds
{42,43,44}, 3B timesteps, everything else exactly as
`scripts/adaptive/cluster_hiddensize_ablation.sh` (minus the hidden-size flags —
default h=256). Tag: `k5_standard_recipe_e010`. NOTE: k=5 context = 1005 tokens
→ larger memory than k=4; you may need to shrink batch further. 3 GPUs, ~2 days.

**Task B — h=512 capacity point (4th GPU while A runs).**
The hidden-size ablation h=512 OOM'd on H100 at full batch (37.7 GiB single
alloc). On 32 GB use: hidden 512 via `--policy.hidden-size 512
--transformer.input-size 512 --transformer.hidden-size 512`, k=4, e_ub 0.10,
partner 2e029h15, seed 42 first (add 43/44 if it fits), minibatch ≤ 12×HORIZON,
vec 16/16/16. Tag: `hidden_size_ablation_k4_e010` (same family tag as the
existing ablation). If it OOMs even at that, report and drop.

**Task C — extra seeds for headline cells (after A/B).**
Seeds 45, 46 for 0.10/k4 and 0.20/k4 (partners 2e029h15 / m2ygolog), standard
recipe. Tightens the error bars on the two headline cells. Tags: reuse the
respective grid tags.

**Evals:** if you have the eval data transferred, run
`eval_final_540.py --wid <wid> --k <k> --seed <s> --iter <last>` per finished
run (540 maps × 20 rollouts fits easily in 32 GB). Otherwise ship checkpoints
back to the cluster and the primary agent evals them.

## 6. Pitfalls (all learned the hard way — do not rediscover)

1. **PyTorch MHA fastpath crashes** (CUDA illegal memory access) at the
   co-player encoder shape. Fixed globally in `pufferlib/pufferl.py` (top:
   `torch.backends.mha.set_fastpath_enabled(False)`). Don't remove it.
2. **`PUFFER_TRANSFORMER_LEGACY_EVAL=1` always.** The streaming KV eval path
   breaks production even with garbage_mask.
3. **In-training eval subprocess** inherits hidden-size/demo flags via
   `pufferlib/utils.py` (fixed 2026-07-03) — if you change model shape flags,
   they now propagate; before that fix every non-h256 run silently lost its
   in-training evals.
4. **total_timesteps must be ≥ a few batches** or `train()` div-by-zero at
   epochs=0 (batch ≈ num_ego × context_len × workers ≈ 13M at k=4/vec32).
5. **sacct lies about renders**: render teardown segfaults after success;
   trust artifact counts, not exit codes. Slurm TIMEOUT can also hit in
   post-training teardown AFTER the final checkpoint is written — check
   `experiments/puffer_adaptive_drive_<wid>/model_*.pt` for the target iter.
6. **Trial-0 demo mode (`--env.demo-trial-0 True`)** exists env-side
   (drive.h `move_expert` snap). Demo-TRAINING is known-harmful (imitation
   collapse + trial-0 PPO gradient contamination — no loss mask); do not launch
   demo-training runs.
7. Resume: `--load-model-path <ckpt>` restores weights+Adam+step via sibling
   `trainer_state.pt`; add `--load-id <wid>` ONLY when continuing the same
   experiment (same run), never when branching two arms from one parent
   (wandb history + exp-dir collisions).
8. Slow/hung nodes happen; treat as one-offs (kill + relaunch manually), don't
   build auto-restart watchers.

## 7. Division of labor / do not touch

- The cluster agent handles: memory-ablation control (in flight), all analysis
  and figures, the paper doc, and all evals unless you have the data locally.
- Don't re-run: demo training, curriculum continuations, e0001 anchor, the
  original grid. All done — results in `notes/paper_analysis.md`.
- Use ONLY the tags specified above so the wandb namespace stays clean.
- When a run finishes: post the wid + final iter in wandb notes (or tell the
  user), so the cluster side can eval/integrate.
