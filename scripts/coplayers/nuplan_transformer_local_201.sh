#!/bin/bash
set -e

# Local 4-GPU launcher for NuPlan Transformer co-players at 201-frame
# trajectories (vs the legacy 91-frame in nuplan_transformer_local.sh).
#
# Why 201: nuplan source scenes are 150-201 frames long. The previous
# 91-frame binaries truncated ~55% of the data on average. Regenerated
# binaries live at resources/drive/binaries/nuplan_201/ (5401 maps).
#
# Grid: e ∈ {0.01, 0.10} × d ∈ {0.8, 0.6}  — same 4 slots as the 91-frame
# script, on GPUs 0-3 so the running k=3 adaptive jobs on 4-7 are
# undisturbed. Override GPUs with: GPUS="0 1 2 3" bash <script>
#
# Stop everything: tmux kill-session -t coplayer_nuplan_tfm_local_201

ENTROPY_UB=(0.01 0.10 0.01 0.10)
DISCOUNT_LB=(0.8  0.8  0.6  0.6)

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=5000
SEED=42

CONTEXT_LENGTH=201
SCENARIO_LENGTH=201
# 201 * 160 = 32160 (close to the legacy 32760 = 91 * 360, but divisible by
# the new horizon — pufferl asserts minibatch_size % horizon == 0).
MINIBATCH_SIZE=32160

GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#ENTROPY_UB[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than grid slots (${#ENTROPY_UB[@]})" >&2
  exit 1
fi
echo "Launching $N_RUNS of ${#ENTROPY_UB[@]} grid slots on GPUs: ${GPU_ARR[*]}"

SESSION=coplayer_nuplan_tfm_local_201
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  EUB=${ENTROPY_UB[$i]}
  DLB=${DISCOUNT_LB[$i]}
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="coplayer_nuplan_tfm_local_201_e${EUB}_d${DLB}_c${COLLISION_LB}_o${OFFROAD_LB}_lane${LANE_REWARD}"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: entropy_ub=$EUB discount_lb=$DLB on GPU $GPU (201-frame)' && \
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
  --env.conditioning.discount-weight-lb $DLB \
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

echo "Launched $N_RUNS 201-frame co-player runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
