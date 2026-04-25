#!/bin/bash
#SBATCH --job-name=coplayer_ablation
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

# Train every co-player ablation variant in one slurm array.
#
#   sbatch scripts/ablations/all_coplayers.sh
#
# 8 array tasks = 2 datasets × 2 archs × 2 conditioning types:
#   idx | dataset | architecture | conditioning
#   ----+---------+--------------+-------------
#    0  | womd    | Recurrent    | none
#    1  | womd    | Recurrent    | all
#    2  | womd    | Transformer  | none
#    3  | womd    | Transformer  | all
#    4  | nuplan  | Recurrent    | none
#    5  | nuplan  | Recurrent    | all
#    6  | nuplan  | Transformer  | none
#    7  | nuplan  | Transformer  | all
#
# Resulting checkpoints land at experiments/puffer_drive_<wandb_run_id>.pt.
# Once they've trained, plug them into scripts/adaptive/*.sh as ZIPPED_RUNS
# entries.

# Decode array index → (dataset, arch, cond_type)
RUNS=(
  "womd Recurrent none"
  "womd Recurrent all"
  "womd Transformer none"
  "womd Transformer all"
  "nuplan Recurrent none"
  "nuplan Recurrent all"
  "nuplan Transformer none"
  "nuplan Transformer all"
)

read -r DATASET ARCH COND <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

# Dataset → map_dir + num_maps
case "$DATASET" in
  nuplan)
    MAP_DIR="resources/drive/binaries/nuplan"
    NUM_MAPS=5000
    ;;
  womd)
    MAP_DIR="resources/drive/binaries/training"
    NUM_MAPS=10000
    ;;
  *) echo "unknown dataset $DATASET" >&2; exit 1 ;;
esac

# Conditioning → puffer flags
# `all` sweeps entropy 0→0.1 and discount 0.8→1.0 (the same range used in
# scripts/coplayers/* with the cell at idx 0).
case "$COND" in
  none)
    COND_ARGS="--env.conditioning.type none"
    ;;
  all)
    COND_ARGS="--env.conditioning.type all \
               --env.conditioning.entropy-weight-lb 0 \
               --env.conditioning.entropy-weight-ub 0.1 \
               --env.conditioning.discount-weight-lb 0.8 \
               --env.conditioning.discount-weight-ub 1.0"
    ;;
  *) echo "unknown conditioning $COND" >&2; exit 1 ;;
esac

TAG="coplayer_${DATASET}_${ARCH,,}_cond-${COND}"

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e

   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate

   nice -n 19 python scripts/gpu_heartbeat.py &
   HEARTBEAT_PID=\$!

   puffer train puffer_drive --wandb \
     --wandb-project ada_coplayer_ablation \
     --tag $TAG \
     --policy-architecture $ARCH \
     --rnn-name $ARCH \
     --env.map-dir $MAP_DIR \
     --env.num-maps $NUM_MAPS \
     --eval.map-dir $MAP_DIR \
     --train.checkpoint-interval 50 \
     $COND_ARGS

   kill \$HEARTBEAT_PID
 "
