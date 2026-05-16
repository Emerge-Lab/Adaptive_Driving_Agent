#!/bin/bash
#SBATCH --job-name=ada_k4_smoke
#SBATCH --output=/scratch/mmk9418/logs/smoke_%j.out
#SBATCH --error=/scratch/mmk9418/logs/smoke_%j.err
#SBATCH --mem=48GB
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Smoke test for cluster_nuplan_transformer_k4_gb3.sh — runs a tiny single-partner
# training to validate:
#   - singularity image + overlay + venv activate
#   - C extension built for the cluster arch
#   - nuplan_201 + nuplan_hard binaries present
#   - partner checkpoint loads
#   - co-player + ego forward passes work on GPU
#   - wandb sync online
#   - first dashboard frame renders without OOM/NaN
#
# Lightweight: nw=4, 5M timesteps (~3-4 min). Submit, watch the log, kill once
# you see Steps advancing.

singularity exec --nv --overlay "$OVERLAY_FILE:ro" "$SINGULARITY_IMAGE" bash -c "
  set -e
  cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
  source .venv/bin/activate
  export WANDB_MODE=online
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  xvfb-run -a puffer train puffer_adaptive_drive \
    --wandb --wandb-project adaptive_aligned \
    --tag smoke_k4_gb3_\$(date +%s) \
    --policy-architecture Transformer --rnn-name Transformer \
    --train.horizon 804 \
    --train.minibatch-multiplier 25 \
    --train.max-minibatch-size 20100 \
    --train.cpu-offload True \
    --train.checkpoint-interval 100 \
    --train.render-interval 100 \
    --train.seed 42 \
    --train.total-timesteps 5000000 \
    --vec.num-workers 4 --vec.num-envs 4 \
    --env.map-dir resources/drive/binaries/nuplan_201 \
    --env.num-maps 4999 \
    --env.scenario-length 201 \
    --env.k-scenarios 4 \
    --env.goal-behavior 3 \
    --env.conditioning.type none \
    --env.reward-lane-align 0.025 \
    --env.co-player-enabled 1 \
    --env.co-player-policy.policy-path experiments/puffer_drive_miku2puk.pt \
    --env.co-player-policy.architecture Transformer \
    --env.co-player-policy.transformer.horizon 201 \
    --env.co-player-policy.conditioning.type all \
    --env.co-player-policy.conditioning.collision-weight-lb -2 \
    --env.co-player-policy.conditioning.collision-weight-ub 0 \
    --env.co-player-policy.conditioning.offroad-weight-lb -2 \
    --env.co-player-policy.conditioning.offroad-weight-ub 0 \
    --env.co-player-policy.conditioning.entropy-weight-lb 0 \
    --env.co-player-policy.conditioning.entropy-weight-ub 0.05 \
    --env.co-player-policy.conditioning.discount-weight-lb 0.4 \
    --env.co-player-policy.conditioning.discount-weight-ub 1 \
    --env.external-co-player-actions True \
    --env.map-rand-per-scenario False \
    --eval.map-dir resources/drive/binaries/nuplan_hard \
    --eval.num-maps 16 \
    --eval.eval-interval 1000
"
