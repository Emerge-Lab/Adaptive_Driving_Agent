#!/bin/bash
#SBATCH --job-name=puffer_baseline
#SBATCH --output=/scratch/mmk9418/logs/%A_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

# Vanilla baseline training script
# No co-players - other vehicles follow recorded human trajectories from Waymo data
# For comparison with co-player experiments in run.sh

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

   puffer train puffer_adaptive_drive --wandb --tag adaptive_baseline \
     --env.num-maps 1000 \
     --env.conditioning.type none \
     --env.co-player-enabled 0 \
     --train.seed 42

   kill \$HEARTBEAT_PID
 "
