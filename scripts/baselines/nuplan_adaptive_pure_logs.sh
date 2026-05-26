#!/bin/bash
#SBATCH --job-name=adaptive_pure_logs
#SBATCH --output=/scratch/mmk9418/logs/%A_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

# Adaptive ego (k=2) trained WITHOUT a coplayer — partners are pure logged
# trajectories during training, same as during human-replay eval.
#
# This is the ceiling test for the headline claim:
#   - If Δ-score on human-replay eval > 0 here, then positive Δ is achievable
#     with the right training partners. The hybrid coplayer+log approach has
#     a meaningful target.
#   - If Δ ≈ 0 (or negative) even here, the human-replay eval format itself
#     limits the room for adaptation (s0 and s1 are deterministic replays of
#     the same logs, so there's not much partner-related signal to exploit).
#     We'd need a different eval protocol (e.g., held-out coplayers).
#
#   sbatch scripts/baselines/nuplan_adaptive_pure_logs.sh

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

   xvfb-run -a -s '-screen 0 1280x720x24' \
   puffer train puffer_adaptive_drive --wandb \
     --wandb-project adaptive_aligned \
     --tag adaptive_pure_logs \
     --policy-architecture Transformer \
     --rnn-name Transformer \
     --train.horizon 182 \
     --train.checkpoint-interval 25 \
     --train.render-interval 25 \
     --train.seed 42 \
     --env.map-dir resources/drive/binaries/nuplan \
     --env.num-maps $NUPLAN_NUM_MAPS \
     --env.k-scenarios 2 \
     --env.conditioning.type none \
     --env.reward-lane-align 0.01 \
     --env.co-player-enabled 0 \
     --eval.map-dir resources/drive/binaries/nuplan \
     --eval.human-replay-eval True \
     --eval.eval-interval 25

   kill \$HEARTBEAT_PID
 "
