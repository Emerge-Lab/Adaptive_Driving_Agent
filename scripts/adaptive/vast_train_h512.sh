#!/bin/bash
# Task B (handoff §5): h=512 capacity point of the hidden-size ablation
# (h512 OOM'd on H100 at full batch — 37.7 GiB single alloc).
# Same family recipe/tag as cluster_hiddensize_ablation.sh, k=4, e_ub=0.10,
# partner 2e029h15; VRAM-shrunk for RTX 5090 32 GB: vec 16/16/16,
# max-minibatch-size MB_MULT x HORIZON (handoff suggests <= 12).
# Usage: vast_train_h512.sh <gpu> <seed> [mb_mult=12] [total_timesteps=3000000000]
set -e
source /venv/main/bin/activate
cd /workspace/Adaptive_Driving_Agent

GPU=$1; SEED=$2; MAX_MINIBATCH_SIZE=${3:-4020}; TOTAL_TIMESTEPS=${4:-3000000000}; EVAL_INTERVAL=${5:-10}; VEC=${6:-16}  # must divide 50*HORIZON=40200: {10050,8040,5025,...}
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PUFFER_TRANSFORMER_LEGACY_EVAL=1

HIDDEN_SIZE=512
PARTNER_ID=2e029h15
ENTROPY_UB=0.10
K_SCENARIOS=4
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))   # 804

TAG="hidden_size_ablation_k4_e010"

echo "[vast_h512] gpu=$GPU seed=$SEED hidden=$HIDDEN_SIZE horizon=$HORIZON max_minibatch=$MAX_MINIBATCH_SIZE steps=$TOTAL_TIMESTEPS tag=$TAG"

xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned_v2 \
  --tag $TAG \
  --policy-architecture Transformer --rnn-name Transformer \
  --policy.hidden-size $HIDDEN_SIZE \
  --transformer.input-size $HIDDEN_SIZE \
  --transformer.hidden-size $HIDDEN_SIZE \
  --train.gamma 0.995 \
  --train.learning-rate 3e-3 \
  --train.ent-coef 0.005 \
  --train.horizon $HORIZON \
  --train.minibatch-multiplier 50 \
  --train.max-minibatch-size $MAX_MINIBATCH_SIZE \
  --train.cpu-offload True \
  --train.checkpoint-interval 10 \
  --train.render-interval 30 \
  --train.seed $SEED \
  --train.total-timesteps $TOTAL_TIMESTEPS \
  --vec.num-workers $VEC --vec.num-envs $VEC --vec.batch-size $VEC \
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps 4999 \
  --env.scenario-length $SCENARIO_LENGTH \
  --env.k-scenarios $K_SCENARIOS \
  --env.goal-behavior 3 \
  --env.conditioning.type none \
  --env.reward-lane-align 0.05 \
  --env.reward-vehicle-collision -0.5 \
  --env.reward-offroad-collision -0.5 \
  --env.co-player-enabled 1 \
  --env.co-player-policy.policy-path experiments/puffer_drive_${PARTNER_ID}.pt \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.transformer.horizon $SCENARIO_LENGTH \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.collision-weight-lb -2 \
  --env.co-player-policy.conditioning.collision-weight-ub 0 \
  --env.co-player-policy.conditioning.offroad-weight-lb -2 \
  --env.co-player-policy.conditioning.offroad-weight-ub 0 \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub $ENTROPY_UB \
  --env.co-player-policy.conditioning.discount-weight-lb 0.4 \
  --env.co-player-policy.conditioning.discount-weight-ub 1 \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --env.entropy-curriculum-enabled False \
  --eval.map-dir              resources/drive/binaries/nuplan_hard \
  --eval.num-maps             540 \
  --eval.human-replay-eval    True \
  --eval.human-replay-num-rollouts 10 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540 \
  --eval.eval-interval $EVAL_INTERVAL
