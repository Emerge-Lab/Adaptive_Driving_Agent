#!/bin/bash
#SBATCH --job-name=baseline_womd_tfm
#SBATCH --output=/scratch/mmk9418/logs/%A_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

# Vanilla baseline on WOMD with Transformer architecture
# No co-players - other vehicles follow recorded human trajectories

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

   puffer train puffer_adaptive_drive --wandb --tag baseline_womd_transformer \
     --env.num-maps 10000 \
     --env.conditioning.type none \
     --env.co-player-enabled 0 \
     --train.seed 42 \
     --rnn-name Transformer

   kill \$HEARTBEAT_PID
 "
