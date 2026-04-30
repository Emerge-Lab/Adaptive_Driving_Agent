#!/bin/bash
set -e

# Co-player training: entropy_ub sweep at fixed discount_lb=0.4.
# Same 201-frame setup as nuplan_transformer_local_201.sh, but the partner
# pool is widened to give us a max-diversity stable to use for diverse-partner
# adaptive ego training. Entropy_ub spans 0.05 → 1.0 in 5 steps, covering
# from "near-deterministic" to "very stochastic" partners.
#
# Why this sweep: prior attention probe found cross-scenario attention scales
# with partner entropy. Higher entropy ⇒ harder-to-predict partner ⇒ ego must
# rely on cache content. We need a partner set spanning that whole range.
#
# Defaults to GPUs 1-5 (GPU 0 is busy with city-adapt run as of 2026-04-29).
# Override with: GPUS="1 2 3 4 5" bash <script>
#
# Stop everything: tmux kill-session -t coplayer_nuplan_tfm_local_201_esweep

ENTROPY_UB=(0.05 0.10 0.20 0.50 1.00)

DISCOUNT_LB=0.4
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=5000
SEED=42

CONTEXT_LENGTH=201
SCENARIO_LENGTH=201
MINIBATCH_SIZE=32160

GPUS=${GPUS:-"1 2 3 4 5"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#ENTROPY_UB[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than entropy_ub slots (${#ENTROPY_UB[@]})" >&2
  exit 1
fi
echo "Launching $N_RUNS of ${#ENTROPY_UB[@]} entropy_ub values on GPUs: ${GPU_ARR[*]}"

SESSION=coplayer_nuplan_tfm_local_201_esweep
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  EUB=${ENTROPY_UB[$i]}
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="coplayer_nuplan_201_esweep_e${EUB}_d${DISCOUNT_LB}_c${COLLISION_LB}_o${OFFROAD_LB}_lane${LANE_REWARD}"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: entropy_ub=$EUB discount_lb=$DISCOUNT_LB on GPU $GPU (201-frame esweep)' && \
xvfb-run -a puffer train puffer_drive \
  --wandb --wandb-project ada_new_coplayers \
  --tag $TAG \
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.scenario-length $SCENARIO_LENGTH \
  --env.reward-lane-align $LANE_REWARD \
  --env.conditioning.type all \
  --env.conditioning.collision-weight-lb $COLLISION_LB \
  --env.conditioning.collision-weight-ub 0 \
  --env.conditioning.offroad-weight-lb $OFFROAD_LB \
  --env.conditioning.offroad-weight-ub 0 \
  --env.conditioning.entropy-weight-lb 0 \
  --env.conditioning.entropy-weight-ub $EUB \
  --env.conditioning.discount-weight-lb $DISCOUNT_LB \
  --env.conditioning.discount-weight-ub $DISCOUNT_UB \
  --policy-architecture Transformer \
  --train.context-length $CONTEXT_LENGTH \
  --train.horizon $CONTEXT_LENGTH \
  --train.minibatch-size $MINIBATCH_SIZE \
  --train.max-minibatch-size $MINIBATCH_SIZE \
  --train.learning-rate 0.003 \
  --train.checkpoint-interval 50 \
  --train.seed $SEED \
  --eval.map-dir resources/drive/binaries/nuplan_201"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS 201-frame entropy-sweep co-player runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "Sweep: entropy_ub ∈ {${ENTROPY_UB[*]}}, discount_lb=${DISCOUNT_LB}"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
