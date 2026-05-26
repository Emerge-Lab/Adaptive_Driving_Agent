#!/bin/bash
# Helper to run a short trial-debug comparison: one gb=0 + one gb=3.
# Each writes PUFFER_TRIAL_DEBUG_FILE for offline diff.
#
# Usage: GOAL_BEHAVIOR=0 ./run_trial_debug_short.sh
#        GOAL_BEHAVIOR=3 ./run_trial_debug_short.sh
#
# Designed for interactive single-GPU runs. Small (nw=4, nv=4) to be cheap.
set -e

GOAL_BEHAVIOR=${GOAL_BEHAVIOR:-3}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-50000000}
SEED=${SEED:-42}
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

DEBUG_DIR=/scratch/mmk9418/projects/Adaptive_Driving_Agent/logs/trial_debug
mkdir -p "$DEBUG_DIR"
DEBUG_FILE="$DEBUG_DIR/gb${GOAL_BEHAVIOR}_seed${SEED}_$(date +%H%M%S).jsonl"

TAG="trial_debug_gb${GOAL_BEHAVIOR}"

export PUFFER_TRIAL_DEBUG_FILE="$DEBUG_FILE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled   # short diagnostic — no wandb noise
# Use legacy full-context eval forward to match train's computation pattern.
export PUFFER_TRANSFORMER_LEGACY_EVAL=1

echo "[trial_debug] gb=$GOAL_BEHAVIOR  seed=$SEED  total_timesteps=$TOTAL_TIMESTEPS"
echo "[trial_debug] debug file = $DEBUG_FILE"

xvfb-run -a puffer train puffer_adaptive_drive \
  --tag "$TAG" \
  --policy-architecture Transformer --rnn-name Transformer \
  --train.gamma "$GAMMA" \
  --train.learning-rate "$LR" \
  --train.ent-coef "$ENT_COEF" \
  --train.horizon "$HORIZON" \
  --train.minibatch-multiplier 50 \
  --train.max-minibatch-size 20100 \
  --train.cpu-offload True \
  --train.checkpoint-interval 99999 \
  --train.render-interval 99999 \
  --train.seed "$SEED" \
  --train.total-timesteps "$TOTAL_TIMESTEPS" \
  --vec.num-workers "$NUM_WORKERS" --vec.num-envs "$NUM_ENVS" --vec.batch-size 4 \
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps "$NUPLAN_NUM_MAPS" \
  --env.scenario-length "$SCENARIO_LENGTH" \
  --env.k-scenarios "$K_SCENARIOS" \
  --env.goal-behavior "$GOAL_BEHAVIOR" \
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
  --eval.eval-interval 99999

echo "[trial_debug] DONE. Records:"
wc -l "$DEBUG_FILE"
