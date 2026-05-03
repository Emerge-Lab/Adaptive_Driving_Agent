#!/bin/bash
set -e

# Track 2 of clipfrac investigation: prio_alpha=0 + ent_coef=0.001.
#
# Track 1 (prio_test, GPU 6) showed prio sampling caused ~30% of the
# clipfrac elevation: baseline (prio=0.85) plateaued at 0.18, prio=0
# settled at 0.13. Hypothesis for the remaining 0.13: ent_coef=0.005
# in adaptive.ini is 5× the default (0.001), adding exploration noise
# to the policy gradient and inflating per-step variance → more clipping
# events.
#
# This run combines both fixes: prio=0 AND ent_coef=0.001. Direct A/B vs
# Track 1 (prio_test, prio=0 + ent=0.005). If clipfrac drops to 0.05-0.10,
# ent_coef was the second contributor and we have a permanent fix.
# If unchanged at 0.13, the residual is structural (bf16 / batch
# correlation from 512 ego agents) and 0.13 is our floor — accept it
# and re-evaluate ada_delta with that.
#
# Default GPU: 7 (gamma sweep on 0-2; prio_test on 6; 7 idle).

LABEL="prio0_ent0001_g0995"
GAMMA=0.995
PRIO_ALPHA=0
PRIO_BETA0=0
ENT_COEF=0.001
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

GPUS=${GPUS:-"7"}
GPU=${GPUS%% *}

SESSION=adaptive_local_k2_201_prio_ent_test
TAG="ada_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp0_${LABEL//./}"

CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: prio_alpha=${PRIO_ALPHA} ent_coef=${ENT_COEF} gamma=${GAMMA} on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.prio-alpha $PRIO_ALPHA \
  --train.prio-beta0 $PRIO_BETA0 \
  --train.ent-coef $ENT_COEF \
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
echo "Launched track-2 ablation in tmux session '$SESSION' on GPU $GPU"
echo "  ${LABEL}: prio=0 + ent=${ENT_COEF} (was 0.005)"
echo "  A/B vs prio_test (prio=0 + ent=0.005, currently at clipfrac ~0.13)"
echo "Attach: tmux attach -t $SESSION"
echo "Stop:   tmux kill-session -t $SESSION"
