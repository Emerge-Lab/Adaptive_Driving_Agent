#!/bin/bash
#SBATCH --job-name=baseline_aligned
#SBATCH --output=/scratch/mmk9418/logs/%A_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

# Non-adaptive baseline that matches the adaptive_aligned env (lane reward
# enabled, nuPlan, Transformer). Used as the floor for interpreting the Δ-score
# of the adaptive_vs_*.sh runs: gives "what does an ego score on human-replay
# without bothering with adaptation?"
#
# Eval is on human-replay every 50 epochs so we can read off the same metric
# the adaptive runs report.
#
#   sbatch scripts/baselines/nuplan_aligned_baseline.sh

NUPLAN_NUM_MAPS=4999

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
     --wandb-project adaptive_aligned \
     --tag baseline_nuplan_lane0.01 \
     --policy-architecture Transformer \
     --rnn-name Transformer \
     --train.context-length 91 \
     --train.horizon 91 \
     --train.checkpoint-interval 50 \
     --train.seed 42 \
     --env.map-dir resources/drive/binaries/nuplan \
     --env.num-maps $NUPLAN_NUM_MAPS \
     --env.conditioning.type none \
     --env.reward-lane-align 0.01 \
     --env.co-player-enabled 0 \
     --eval.map-dir resources/drive/binaries/nuplan \
     --eval.human-replay-eval True \
     --eval.eval-interval 50

   kill \$HEARTBEAT_PID
 "
