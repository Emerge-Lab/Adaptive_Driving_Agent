#!/bin/bash
set -e

# Oracle ablation (γ=0.995, partner 2e029h15).
#
# Same config as gamma sweep run 4lm6kkh7 (γ=0.995, partner 2e029h15),
# but with --env.ego-is-oracle True. The ego now sees the partner's
# 5-dim conditioning vector appended to its obs.
#
# Why: clipfrac investigation showed prio=0 + clip=0.3 brought clipfrac
# from 0.18 → 0.08, but ada_delta stayed at ±0.005 across 23 epochs.
# Attention probe showed the model attends ~0.65 mass on past scenario
# slots — so attention IS working. The bottleneck is downstream.
#
# Two possibilities:
#   1. Architecture cannot translate partner-info into adapted behavior
#   2. Architecture works, but in-context inference of partner type fails
#
# Oracle hands the partner type directly. If ada_delta jumps → (1) is
# false, problem is in-context inference (next: longer context window,
# explicit partner ID embedding, etc). If ada_delta still ~0 → (1) is
# true, action conditioning or reward-shape is the bottleneck.
#
# Default GPU: 3 (gamma sweep on 0-2; track1 on 6; track3 on 7; 4-5 idle).

LABEL="oracle_g0995_e01"
GAMMA=0.995
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
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))

NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"3"}
GPU=${GPUS%% *}

SESSION=adaptive_local_k2_201_oracle_g0995
TAG="ada_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp0_${LABEL//./}"

CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: ego_is_oracle=True gamma=${GAMMA} on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.gamma $GAMMA \
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
  --env.ego-is-oracle True \
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
echo "Launched ORACLE ablation in tmux session '$SESSION' on GPU $GPU"
echo "  ${LABEL}: ego_is_oracle=True (sees partner's 5-dim conditioning vector)"
echo "  Direct A/B vs gamma sweep run 4lm6kkh7 (same config, ego_is_oracle=False)"
echo "  Watching ada_delta — should be > 0 if architecture can use partner info"
echo "Attach: tmux attach -t $SESSION"
echo "Stop:   tmux kill-session -t $SESSION"
