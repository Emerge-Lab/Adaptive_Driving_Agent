#!/bin/bash
#SBATCH --job-name=puffer_drive
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

# Define configurations for each array task ID
# Each entry: "path entropy_weight_ub discount_weight_lb"
ZIPPED_RUNS=(
  "experiments/puffer_drive_8u92j3ts/model_puffer_drive_003000.pt 0.5 0.8"
  "experiments/puffer_drive_x4xs711x.pt 0.1 0.8"
  "experiments/puffer_drive_hhzdzhl8.pt 0.01 0.8"
  "experiments/puffer_drive_3xd48djp.pt 0 0.8"

  "experiments/puffer_drive_fgglgofu.pt 0.5 0.6"
  "experiments/puffer_drive_g3x9e5rn.pt 0.01 0.6"
  "experiments/puffer_drive_gzuuzs0o.pt 0.1 0.6"
  "experiments/puffer_drive_6nzf7xha.pt 0 0.6"

  "experiments/puffer_drive_3iefv59j.pt 0.5 0.4"
  "experiments/puffer_drive_7h07nrxy.pt 0.1 0.4"
  "experiments/puffer_drive_bot2wl0m.pt 0.01 0.4"
  "experiments/puffer_drive_n7mx9f4b.pt 0 0.4"

  "experiments/puffer_drive_9jv4q77m.pt 0.5 0.2"
  "experiments/puffer_drive_5p8gpw84.pt 0.1 0.2"
  "experiments/puffer_drive_jskw659g.pt 0.01 0.2"
  "experiments/puffer_drive_eeyizdrk.pt 0 0.2"
)

# Parse the values for this array task
read -r COPLAYER_PATH ENTROPY_UB DISCOUNT_LB <<< "${ZIPPED_RUNS[$SLURM_ARRAY_TASK_ID]}"

# Fixed values
CONDITION_TYPE="all"
DISCOUNT_UB=1
ENTROPY_LB=0

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e

   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate

   # Start GPU heartbeat in background (for RL training which is CPU-bound)
   nice -n 19 python scripts/gpu_heartbeat.py &
   HEARTBEAT_PID=\$!
   echo \"Started GPU Heartbeat with PID: \$HEARTBEAT_PID\"

   puffer train puffer_adaptive_drive --wandb --env.num-maps 1000 \
     --env.conditioning.type none \
     --env.co-player-enabled 1 \
     --env.co-player-policy.policy-path $COPLAYER_PATH \
     --env.co-player-policy.conditioning.type $CONDITION_TYPE \
     --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
     --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
     --env.co-player-policy.conditioning.entropy-weight-lb $ENTROPY_LB \
     --env.co-player-policy.conditioning.entropy-weight-ub $ENTROPY_UB

   kill \$HEARTBEAT_PID
 "
