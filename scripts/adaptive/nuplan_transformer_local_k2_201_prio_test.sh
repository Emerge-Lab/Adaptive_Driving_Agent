#!/bin/bash
set -e

# Priority-sampling ablation. Default config inherits prio_alpha=0.85,
# prio_beta0=0.85 from upstream PufferDrive. Hypothesis: with our long
# horizon (402 vs upstream's 32), high-|adv| segments get over-sampled
# across PPO minibatches, the importance ratio drifts as those segments
# are revisited within an epoch, and we get the clipfrac plateau (~0.17)
# that we cannot move with LR or optimizer changes.
#
# This run: prio_alpha=0, prio_beta0=0 (uniform sampling, no IS correction).
# Direct A/B against gamma sweep run 4lm6kkh7 (γ=0.995, partner 2e029h15,
# everything else identical, prio_alpha=0.85). If clipfrac drops to
# 0.08-0.12, prioritized sampling is the culprit. If unchanged, move
# to bptt_horizon next.
#
# Default GPU: 6 (gamma sweep e01 runs on 0-2; 3-7 idle after e05 kill).

LABEL="prio0_g0995_e01"
GAMMA=0.995
PRIO_ALPHA=0
PRIO_BETA0=0
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

SESSION=adaptive_local_k2_201_prio_test
TAG="ada_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp0_${LABEL//./}"

CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: prio_alpha=${PRIO_ALPHA} prio_beta0=${PRIO_BETA0} gamma=${GAMMA} on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.prio-alpha $PRIO_ALPHA \
  --train.prio-beta0 $PRIO_BETA0 \
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
echo "Launched priority-sampling ablation in tmux session '$SESSION' on GPU $GPU"
echo "  ${LABEL}: prio_alpha=${PRIO_ALPHA} prio_beta0=${PRIO_BETA0} (was 0.85/0.85)"
echo "  A/B vs 4lm6kkh7 (γ=0.995, partner 2e029h15, prio=0.85/0.85)"
echo "  Watching for: clipfrac trajectory + approx_kl + value_loss"
echo "Attach: tmux attach -t $SESSION"
echo "Stop:   tmux kill-session -t $SESSION"
