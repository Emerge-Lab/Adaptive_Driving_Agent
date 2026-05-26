#!/bin/bash
#SBATCH --job-name=eval_k4_gb3
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --array=0-34

# Offline human-replay eval over the k=4 gb=3 sweep.
#   35 wandb runs, each task evaluates ONE wid across ALL 8 checkpoints
#   (iter 10,20,30,40,50,60,70,76) and appends offline_eval/human_replay_* to
#   the original wandb run via wandb resume="must". One wandb session per wid,
#   sequential evals inside.
#
# Eval scale: 20 rollouts × 540 maps × 540 agents on nuplan_hard.
# Bumping num_agents 64 → 540 puts one SDC per nuplan_hard map per rollout
# (max_controlled_agents=1, so num_agents IS the parallel-rollout count for
# this env's internal vectorization). Total 20×540 = 10,800 SDC-rollouts;
# ~3.5× per-step throughput vs the 64-agent baseline.
# Estimated ~10-12 min per eval on h100 → ~80-100 min per wid for 8 iters.
#   Per-iter timeout = 1 hr; SLURM wallclock = 6 hr.
#
# nbbbsmyr is intentionally excluded — its experiment dir is missing.
#
# Submit: sbatch scripts/adaptive/cluster_eval_k4_gb3.sh

# 35 wids, 3 groups (A=EntCurriculum, B=lowEgoPenalty, C=per-partner).
WIDS=(
  # Group A — ada_k4_gb3_lowEgoPenalty_EntCurriculum (11; nbbbsmyr missing)
  rwg5a65x icmjygwf 6rv8gcrr
  ipdv2oag nbipb5q9 l9wv41ct
  hke6hyik             3s24do45
  rx3yj0k7 diocrfd9 se7ovksg
  # Group B — ada_k4_gb3_lowEgoPenalty (12)
  uljixs7j 251vz655 0rkojso4
  rmsghbiu h7fajqan wplaas1l
  5obko2iy umaskfka a37ay3nb
  4cowebjw t1bkn7fq s1ro6tzv
  # Group C — per-partner (-2 penalty) (12)
  mpyo1ucm auspoa8z 4tqk602k
  38g805cy he3wmzo4 0da431f2
  o13lmh0q 5fhn4zng 438s7mb2
  96f15g3o 6nrtfrex huk9yuqd
)

WID=${WIDS[$SLURM_ARRAY_TASK_ID]}

# Coordinate with the local 2-parallel driver via shared .done markers in the
# project filesystem. First to finish wins; the other path skips.
DONE_MARKER=logs/offline_eval/${WID}.done
if [ -f "$DONE_MARKER" ]; then
  echo "[cluster_eval_k4_gb3] task=$SLURM_ARRAY_TASK_ID wid=$WID SKIP (done marker exists)"
  exit 0
fi

echo "[cluster_eval_k4_gb3] task=$SLURM_ARRAY_TASK_ID wid=$WID START"

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e

   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate

   export WANDB_MODE=online
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

   xvfb-run -a python scripts/adaptive/eval_k4_gb3_one.py \
     --wid $WID \
     --num-rollouts 20 \
     --num-maps 540 \
     --num-agents 540 \
     --timeout-sec 3600 \
   && touch $DONE_MARKER
 "
