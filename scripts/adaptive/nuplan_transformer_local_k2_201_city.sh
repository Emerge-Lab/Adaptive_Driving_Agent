#!/bin/bash
set -e

# k=2 adaptive log-replay training across nuPlan cities.
# Train on US (Boston + Pittsburgh + Vegas, 5072 scenes), eval on Singapore (329).
# This is "log-replay training": co-player is disabled; non-ego agents follow
# their recorded human trajectories. The ego is the only policy-controlled agent
# per scene (max_controlled_agents=1).
#
# map_rand_per_scenario=False per Eugene/Ed: keep the cache aligned with the
# current scene across the boundary. The k=2 framing here is "continue driving
# in the same scene for 402 frames", split into halves for ada_delta scoring.
# If we want true cross-scene adaptation later, flip --env.map-rand to True.
#
# Default GPUs: 0-3 (4 seeds in parallel).
#
# Override GPUs:    GPUS="0 1 2 3" bash <script>     (default already 0-3)
# Stop everything:  tmux kill-session -t adaptive_local_k2_201_city

SEEDS=(11)

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
NUPLAN_NUM_MAPS_TRAIN=5072
NUPLAN_NUM_MAPS_EVAL=329

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 402

# Same memory profile as the existing k=2/201 launcher (we know this fits at nw=32).
NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"0"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#SEEDS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than seeds (${#SEEDS[@]})" >&2
  exit 1
fi

SESSION=adaptive_local_k2_201_city
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  SEED=${SEEDS[$i]}
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="adaptive_logreplay_k2_201_us2sing_seed${SEED}"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: log-replay k=$K_SCENARIOS, train=us_train, eval=singapore, seed=$SEED on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.horizon $HORIZON \
  --train.minibatch-multiplier $MINIBATCH_MULTIPLIER \
  --train.max-minibatch-size $MAX_MINIBATCH_SIZE \
  --train.cpu-offload True \
  --train.checkpoint-interval 10 \
  --train.render-interval 10 \
  --train.seed $SEED \
  --vec.num-workers $NUM_WORKERS \
  --vec.num-envs $NUM_ENVS \
  --env.map-dir resources/drive/binaries/nuplan_201_us_train \
  --env.num-maps $NUPLAN_NUM_MAPS_TRAIN \
  --env.scenario-length $SCENARIO_LENGTH \
  --env.k-scenarios $K_SCENARIOS \
  --env.conditioning.type none \
  --env.reward-lane-align $LANE_REWARD \
  --env.co-player-enabled 0 \
  --env.map-rand-per-scenario False \
  --eval.map-dir resources/drive/binaries/nuplan_201_singapore \
  --eval.num-maps $NUPLAN_NUM_MAPS_EVAL \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS log-replay city-adapt runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "Train: us_train (5072 scenes from Boston + Pittsburgh + Vegas)"
echo "Eval:  singapore (329 scenes, held-out city)"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
