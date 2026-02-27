#!/bin/bash
#SBATCH --job-name=baseline_nuplan_tfm
#SBATCH --output=/scratch/mmk9418/logs/%A_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

# Vanilla baseline on NuPlan with Transformer architecture
# No co-players - other vehicles follow recorded human trajectories
#
# PREREQUISITE: Convert NuPlan JSON to binary format first:
#   python -c "from pufferlib.ocean.drive.drive import process_all_maps; \
#              process_all_maps('data/nuplan_gpudrive/nuplan', max_maps=5000)"

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

   puffer train puffer_adaptive_drive --wandb --tag baseline_nuplan_transformer \
     --env.map-dir resources/drive/binaries/nuplan \
     --env.num-maps $NUPLAN_NUM_MAPS \
     --env.conditioning.type none \
     --env.co-player-enabled 0 \
     --train.seed 42 \
     --rnn-name Transformer \
     --train.policy-architecture Transformer

   kill \$HEARTBEAT_PID
 "
