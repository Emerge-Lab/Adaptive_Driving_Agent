#!/bin/bash
set -e

# Track 3 of clipfrac investigation: prio_alpha=0 + clip_coef=0.3.
#
# Track 1 (prio_test, GPU 6): prio=0 dropped clipfrac from 0.18 baseline
# to 0.15 — partial fix, then climbed back from 0.12 honeymoon to 0.15.
# Track 2 (prio_ent_test, killed): lower ent_coef (0.005 → 0.001) didn't
# help (early epochs were 0.152, matching Track 1).
#
# Track 3 tests the diagnostic question: is the policy being CLIPPED
# because gradient updates are intrinsically too large (drift), or is
# the clip threshold (0.2) artificially low for our setup?
#
# Raise clip_coef 0.2 → 0.3 (50% wider). Two outcomes:
# - clipfrac drops to ~0.05 → clip threshold WAS the binding constraint;
#   policy wants larger updates than clip allows. We may have been
#   throwing out useful gradient signal — adoption candidate.
# - clipfrac stays at ~0.15 → drift is intrinsic; widening clip just
#   means more events are "non-events". Confirms structural floor.
#
# Direct A/B vs Track 1 (prio=0, clip=0.2). Same all else.
#
# Default GPU: 7 (gamma sweep on 0-2; prio_test on 6; 7 idle).

LABEL="prio0_clip03_g0995"
GAMMA=0.995
PRIO_ALPHA=0
PRIO_BETA0=0
CLIP_COEF=0.3
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

SESSION=adaptive_local_k2_201_prio_clip_test
TAG="ada_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp0_${LABEL//./}"

CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: prio_alpha=${PRIO_ALPHA} clip_coef=${CLIP_COEF} gamma=${GAMMA} on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.prio-alpha $PRIO_ALPHA \
  --train.prio-beta0 $PRIO_BETA0 \
  --train.clip-coef $CLIP_COEF \
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
echo "Launched track-3 ablation in tmux session '$SESSION' on GPU $GPU"
echo "  ${LABEL}: prio=0 + clip_coef=${CLIP_COEF} (was 0.2)"
echo "  A/B vs prio_test (prio=0 + clip=0.2, currently at clipfrac ~0.15)"
echo "  Diagnostic: drop to 0.05 = clip was binding; flat 0.15 = drift is intrinsic"
echo "Attach: tmux attach -t $SESSION"
echo "Stop:   tmux kill-session -t $SESSION"
