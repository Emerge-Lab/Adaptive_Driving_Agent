#!/bin/bash
set -e

# k=2 adaptive ego training — partner sweep across the 5 entropy-conditioned
# partners. Holds γ=0.995 (winner from gamma_sweep) and bumps lane reward to
# 0.025 (up from 0.01) per request. Eval against nuplan_hard.
#
# Goal: isolate partner stochasticity as a driver of ada_delta_score. One ego
# per partner; the only knob that varies between runs is which partner the ego
# trains/evaluates against.
#
# (label, partner_id, entropy_ub) — entropy_ub matches the partner's training
# entropy so the conditioning we sample stays inside its support.
RUNS=(
  "p005_miku2puk  miku2puk  0.05"
  "p010_2e029h15  2e029h15  0.10"
  "p020_m2ygolog  m2ygolog  0.20"
  "p050_6rauydj2  6rauydj2  0.50"
  "p100_n48teqjs  n48teqjs  1.00"
)

GAMMA=0.995
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.025                    # bumped from 0.01
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

# RUN_INDICES selects which entries of RUNS[] to launch (space-separated 0-based
# indices). GPUS is a parallel list — RUN_INDICES[i] runs on GPUS[i].
# Default: all 5 runs on GPUs 0-4.
RUN_INDICES=${RUN_INDICES:-"0 1 2 3 4"}
GPUS=${GPUS:-"0 1 2 3 4"}
read -r -a IDX_ARR <<< "$RUN_INDICES"
read -r -a GPU_ARR <<< "$GPUS"
if [ "${#IDX_ARR[@]}" -ne "${#GPU_ARR[@]}" ]; then
  echo "ERROR: RUN_INDICES count (${#IDX_ARR[@]}) != GPUS count (${#GPU_ARR[@]})" >&2
  exit 1
fi

SESSION=${SESSION:-adaptive_partner_sweep}
tmux has-session -t "$SESSION" 2>/dev/null && echo "Session $SESSION already exists; appending windows" || tmux new-session -d -s "$SESSION" -n placeholder

for ((i=0; i<${#IDX_ARR[@]}; i++)); do
  IDX=${IDX_ARR[$i]}
  GPU=${GPU_ARR[$i]}
  read -r LABEL PARTNER_ID EUB <<< "${RUNS[$IDX]}"
  WIN="${LABEL}"
  TAG="ada_partsweep_${LABEL}_g${GAMMA}_lane${LANE_REWARD}_k2_201"
  COPLAYER="experiments/puffer_drive_${PARTNER_ID}.pt"

  tmux new-window -t "$SESSION" -n "$WIN"

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: partner=${PARTNER_ID} (e_ub=${EUB}) γ=${GAMMA} lane=${LANE_REWARD} on GPU $GPU' && \
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
  --eval.map-dir              resources/drive/binaries/nuplan_hard \
  --eval.num-maps             540 \
  --eval.human-replay-eval    True \
  --eval.human-replay-num-rollouts 200 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540 \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

# kill the placeholder window if it still exists
tmux kill-window -t "$SESSION:placeholder" 2>/dev/null || true

echo
echo "Launched ${#IDX_ARR[@]} partner-sweep runs in tmux session '$SESSION'"
for ((i=0; i<${#IDX_ARR[@]}; i++)); do
  IDX=${IDX_ARR[$i]}
  read -r LABEL PARTNER_ID EUB <<< "${RUNS[$IDX]}"
  echo "  GPU ${GPU_ARR[$i]}: $LABEL  (partner=$PARTNER_ID  e_ub=$EUB)"
done
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
