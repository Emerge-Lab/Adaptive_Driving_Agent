#!/bin/bash
# PLR-lite map-curriculum arms: k4 standard recipe + online priority sampling.
# standard grid recipe (partner 2e029h15, e_ub 0.10, h=256).
# Identical to cluster_hiddensize_ablation.sh recipe (default h=256) except:
#   - K_SCENARIOS=4 (HORIZON=1005)
#   - VRAM-shrunk geometry for RTX 5090 32 GB (handoff §4): vec 16/16/16,
#     max-minibatch-size MB_MULT x HORIZON (tune via smoke test)
# Usage: vast_train_k5.sh <gpu> <seed> [mb_mult=20] [total_timesteps=3000000000]
set -e
source /venv/main/bin/activate
cd /workspace/Adaptive_Driving_Agent

GPU=$1; SEED=$2; CUR_MODE=${7:-success}; MAX_MINIBATCH_SIZE=${3:-8040}; TOTAL_TIMESTEPS=${4:-3000000000}; EVAL_INTERVAL=${5:-10}; VEC=${6:-16}  # must divide 50*HORIZON=40200, multiple of 804: {8040,20100,40200}
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=${WANDB_MODE:-online}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PUFFER_TRANSFORMER_LEGACY_EVAL=1

PARTNER_ID=2e029h15
ENTROPY_UB=0.10
K_SCENARIOS=4
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))   # 804

TAG="plr_map_curriculum_k4_e010"

echo "[vast_plr] mode=$CUR_MODE gpu=$GPU seed=$SEED horizon=$HORIZON max_minibatch=$MAX_MINIBATCH_SIZE steps=$TOTAL_TIMESTEPS tag=$TAG"

xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned_v2 \
  --tag $TAG \
  --policy-architecture Transformer --rnn-name Transformer \
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
  --env.map-curriculum-mode $CUR_MODE \
  --env.entropy-curriculum-enabled False \
  --eval.map-dir              resources/drive/binaries/nuplan_hard \
  --eval.num-maps             540 \
  --eval.human-replay-eval    True \
  --eval.human-replay-num-rollouts 10 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540 \
  --eval.eval-interval $EVAL_INTERVAL
