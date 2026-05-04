#!/bin/bash
set -e

# k=2 adaptive ego training with entropy curriculum, against the e=0.10
# and e=0.20 partners. Tests whether annealing the partner's entropy
# upper-bound during training produces a stronger adaptive ego than
# training at the fixed final value.
#
# Why these two: e=0.10 is where partner_sweep already shows ada_delta
# ≈ +0.155 on hard-map eval — see if curriculum lifts it further. e=0.20
# is the cliff where ada_delta collapses to ~0 — see if curriculum
# rescues it.
#
# Match partner_sweep config (γ=0.995, lane=0.025, conditioning.type=none
# on ego) so results are directly comparable to the +0.155 / -0.010
# numbers from `eval_partner_sweep_m10.log`.
#
# In-training eval uses the M=10 × 540 hard-map recipe (the original
# M=200 timed out, was wasting wall time).
#
# Defaults to GPUs 5 and 6 (DE-grid is on GPUs 0-4).
# Override:    GPUS="5 6" bash <script>
# Stop all:    tmux kill-session -t adaptive_curr_e010_e020

# (label, final_entropy_ub, partner_id)
RUNS=(
  "curr_e010  0.10  2e029h15"
  "curr_e020  0.20  m2ygolog"
)

GAMMA=0.995
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.025
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

GPUS=${GPUS:-"5 6"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#RUNS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than runs (${#RUNS[@]})" >&2
  exit 1
fi

SESSION=adaptive_curr_e010_e020
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r LABEL EUB PARTNER_ID <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}_${LABEL//./}"
  TAG="adaptive_${LABEL}_vs_${PARTNER_ID}_g${GAMMA}_lane${LANE_REWARD}_k2_201"
  COPLAYER="experiments/puffer_drive_${PARTNER_ID}.pt"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t "$SESSION:exp0" "$WIN"
  else
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: curriculum 0 → ${EUB} vs ${PARTNER_ID} γ=${GAMMA} on GPU $GPU' && \
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
  --env.co-player-policy.conditioning.entropy-weight-ub $EUB \
  --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
  --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --env.entropy-curriculum-enabled True \
  --eval.map-dir              resources/drive/binaries/nuplan_hard \
  --eval.num-maps             540 \
  --eval.human-replay-eval    True \
  --eval.human-replay-num-rollouts 10 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540 \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS curriculum runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
for ((i=0; i<N_RUNS; i++)); do
  read -r LABEL EUB PARTNER_ID <<< "${RUNS[$i]}"
  echo "  GPU ${GPU_ARR[$i]}: ${LABEL}  curriculum 0 → ${EUB}  vs ${PARTNER_ID}"
done
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
