#!/bin/bash
#SBATCH --job-name=coplayer_nuplan_rnn
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

# Train co-player policies on NuPlan with Recurrent architecture
# Varies entropy/discount conditioning for diverse co-player behaviors
#
# PREREQUISITE: Convert NuPlan JSON to binary format first:
#   python -c "from pufferlib.ocean.drive.drive import process_all_maps; \
#              process_all_maps('data/nuplan_gpudrive/nuplan', max_maps=5000)"

# Grid: 4 entropy levels × 4 discount levels = 16 configurations
ZIPPED_RUNS=(
  "0.5 0.8"
  "0.1 0.8"
  "0.01 0.8"
  "0 0.8"

  "0.5 0.6"
  "0.1 0.6"
  "0.01 0.6"
  "0 0.6"

  "0.5 0.4"
  "0.1 0.4"
  "0.01 0.4"
  "0 0.4"

  "0.5 0.2"
  "0.1 0.2"
  "0.01 0.2"
  "0 0.2"
)

read -r ENTROPY_UB DISCOUNT_LB <<< "${ZIPPED_RUNS[$SLURM_ARRAY_TASK_ID]}"

# Fixed values
CONDITION_TYPE="all"
DISCOUNT_UB=1
ENTROPY_LB=0
NUPLAN_NUM_MAPS=5000

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

   puffer train puffer_drive --wandb --wandb-project ada_new_coplayers --tag coplayer_nuplan_recurrent \
     --env.map-dir resources/drive/binaries/nuplan \
     --env.num-maps $NUPLAN_NUM_MAPS \
     --env.conditioning.type $CONDITION_TYPE \
     --env.conditioning.entropy-weight-lb $ENTROPY_LB \
     --env.conditioning.entropy-weight-ub $ENTROPY_UB \
     --env.conditioning.discount-weight-lb $DISCOUNT_LB \
     --env.conditioning.discount-weight-ub $DISCOUNT_UB \
     --rnn-name Recurrent \
     --train.checkpoint-interval 50 \
     --eval.map-dir resources/drive/binaries/nuplan

   kill \$HEARTBEAT_PID
 "
