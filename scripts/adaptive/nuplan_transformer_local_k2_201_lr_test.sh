#!/bin/bash
set -e

# Single-run LR ablation. The full gamma sweep showed clipfrac stuck at
# 0.18-0.20 — same as the broken (pre-K/V-cache-fix) runs. Hypothesis:
# LR=0.003 (current default) is high enough that 11 PPO minibatch
# updates per rollout push the policy past the clip threshold, fighting
# the clip rather than learning efficiently. This run drops LR to 1e-4
# (30x lower) to test how much LR contributes.
#
# Other config:
#   - K/V cache fix (commit 5c1ca48c)            ✓ active
#   - Sinusoidal PE (commit eb2d7fe2)             ✓ active
#   - γ=0.995 (the sweet spot from the broken sweep where we saw
#     ada_delta=+0.0028)
#   - Partner = 2e029h15 at e_ub=0.10 (matched, partner showed signal)
#   - All else identical to the gamma sweep
#
# Default GPU: 6 (gamma sweep uses 0-5; 6-7 idle).
#
# Override GPU:    GPUS="6" bash <script>
# Stop:            tmux kill-session -t adaptive_local_k2_201_lr_test

LABEL="lr1e4_g0995_e01"
GAMMA=0.995
LEARNING_RATE=0.0001
PARTNER_ID=2e029h15
COPLAYER=experiments/puffer_drive_${PARTNER_ID}.pt
ENTROPY_UB=0.10

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 402

NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"6"}
GPU=${GPUS%% *}

SESSION=adaptive_local_k2_201_lr_test
TAG="ada_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp0_${LABEL//./}"

CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: gamma=${GAMMA} lr=${LEARNING_RATE} partner=${PARTNER_ID} on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.gamma $GAMMA \
  --train.learning-rate $LEARNING_RATE \
  --train.horizon $HORIZON \
  --train.minibatch-multiplier $MINIBATCH_MULTIPLIER \
  --train.max-minibatch-size $MAX_MINIBATCH_SIZE \
  --train.cpu-offload True \
  --train.checkpoint-interval 10 \
  --train.render-interval 30 \
  --train.seed $SEED \
  --vec.num-workers $NUM_WORKERS \
  --vec.num-envs $NUM_ENVS \
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.scenario-length $SCENARIO_LENGTH \
  --env.k-scenarios $K_SCENARIOS \
  --env.conditioning.type none \
  --env.reward-lane-align $LANE_REWARD \
  --env.co-player-enabled 1 \
  --env.co-player-policy.policy-path $COPLAYER \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.transformer.horizon $SCENARIO_LENGTH \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.collision-weight-lb $COLLISION_LB \
  --env.co-player-policy.conditioning.collision-weight-ub 0 \
  --env.co-player-policy.conditioning.offroad-weight-lb $OFFROAD_LB \
  --env.co-player-policy.conditioning.offroad-weight-ub 0 \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub $ENTROPY_UB \
  --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
  --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.human-replay-num-rollouts 50 \
  --eval.eval-interval 10"

tmux send-keys -t "$SESSION:exp0_${LABEL//./}" "$CMD" C-m

echo
echo "Launched LR-ablation run in tmux session '$SESSION' on GPU $GPU"
echo "  ${LABEL}: gamma=${GAMMA} lr=${LEARNING_RATE} (was 0.003) + K/V cache fix + sinusoidal PE"
echo "Attach: tmux attach -t $SESSION"
echo "Stop:   tmux kill-session -t $SESSION"
