#!/bin/bash
# Run gb=3, k=2, 500M steps on the interactive alloc with wandb + renders + evals.
# Uses the legacy-eval default (models.py default = 1) since that's the fix.
set -e

SEED=${SEED:-42}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-500000000}
LR=${LR:-3e-3}
ENT_COEF=${ENT_COEF:-0.005}

PARTNER_ID=2e029h15
ENTROPY_UB=0.10
COPLAYER_PATH="experiments/puffer_drive_${PARTNER_ID}.pt"

GAMMA=0.995
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.05
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))

NUM_WORKERS=4
NUM_ENVS=4

TAG="alloc_k2_gb3_lefix_500M"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online

echo "[alloc_gb3_500M] seed=$SEED total_timesteps=$TOTAL_TIMESTEPS tag=$TAG"

xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned_v2 \
  --tag "$TAG" \
  --policy-architecture Transformer --rnn-name Transformer \
  --train.gamma "$GAMMA" \
  --train.learning-rate "$LR" \
  --train.ent-coef "$ENT_COEF" \
  --train.horizon "$HORIZON" \
  --train.minibatch-multiplier 50 \
  --train.max-minibatch-size 20100 \
  --train.cpu-offload True \
  --train.checkpoint-interval 20 \
  --train.render-interval 30 \
  --train.seed "$SEED" \
  --train.total-timesteps "$TOTAL_TIMESTEPS" \
  --vec.num-workers "$NUM_WORKERS" --vec.num-envs "$NUM_ENVS" --vec.batch-size 4 \
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps "$NUPLAN_NUM_MAPS" \
  --env.scenario-length "$SCENARIO_LENGTH" \
  --env.k-scenarios "$K_SCENARIOS" \
  --env.goal-behavior 3 \
  --env.conditioning.type none \
  --env.reward-lane-align "$LANE_REWARD" \
  --env.reward-vehicle-collision -0.5 \
  --env.reward-offroad-collision -0.5 \
  --env.co-player-enabled 1 \
  --env.co-player-policy.policy-path "$COPLAYER_PATH" \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.transformer.horizon "$SCENARIO_LENGTH" \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.collision-weight-lb "$COLLISION_LB" \
  --env.co-player-policy.conditioning.collision-weight-ub 0 \
  --env.co-player-policy.conditioning.offroad-weight-lb "$OFFROAD_LB" \
  --env.co-player-policy.conditioning.offroad-weight-ub 0 \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub "$ENTROPY_UB" \
  --env.co-player-policy.conditioning.discount-weight-lb "$DISCOUNT_LB" \
  --env.co-player-policy.conditioning.discount-weight-ub "$DISCOUNT_UB" \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --env.entropy-curriculum-enabled False \
  --eval.map-dir              resources/drive/binaries/nuplan_hard \
  --eval.num-maps             540 \
  --eval.human-replay-eval    True \
  --eval.human-replay-num-rollouts 10 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540 \
  --eval.eval-interval 30

echo "[alloc_gb3_500M] DONE"
