#!/bin/bash
set -e

# Same 4 k=3 adaptive runs as nuplan_transformer_local_k3.sh, but with
# --env.map-rand-per-scenario True so each scenario inside an episode uses
# a fresh map. The ego policy's K/V cache is preserved across scenarios so
# past-scenario context is the only stable signal — forces the policy to
# actually use its in-context memory.
#
# Uses the optimized flags (cpu_offload + external_co_player_actions) so
# we can run nw=32 on a 32 GiB GPU. Same fresh-from-scratch start as the
# k3 launcher (no --load-id; the existing k3 checkpoints were trained
# with map_rand=False so resume would mix two distributions).
#
# Override GPUs:    GPUS="4 5 6 7" bash <script>     (default already 4-7)
# Stop everything:  tmux kill-session -t adaptive_local_k3_maprand

COPLAYERS=(
  "ampggcji 0.10 0.8"
  "q1mtzo02 0.10 0.6"
  "ureqzfe9 0.01 0.6"
  "hntq6ykn 0.01 0.8"
)

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=3
SCENARIO_LENGTH=91
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 273

# With cpu_offload the obs buffer lives in pinned RAM, freeing the GPU. nw=32
# fits easily on a 5090 (peak ~18 GiB). Same as the resume launcher.
NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=200
MAX_MINIBATCH_SIZE=54600

GPUS=${GPUS:-"4 5 6 7"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#COPLAYERS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than coplayer slots (${#COPLAYERS[@]})" >&2
  exit 1
fi

# --- snapshot latest co-player checkpoints (uses 91-frame coplayers, the
# 201-frame ones are still training) ---
echo "Snapshotting latest co-player checkpoints…"
for entry in "${COPLAYERS[@]}"; do
  read -r ID _ _ <<< "$entry"
  src=$(ls /workspace/ADA/experiments/puffer_drive_$ID/model_puffer_drive_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$src" ]; then
    echo "  ERROR: no checkpoint found for puffer_drive_$ID" >&2
    exit 1
  fi
  cp -f "$src" "/workspace/ADA/experiments/puffer_drive_$ID.pt"
  echo "  $ID: $(basename "$src")"
done

SESSION=adaptive_local_k3_maprand
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r ID EUB DLB <<< "${COPLAYERS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="adaptive_vs_${ID}_e${EUB}_d${DLB}_lane${LANE_REWARD}_k${K_SCENARIOS}_maprand"
  CKPT="experiments/puffer_drive_${ID}.pt"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: co-player=$ID (e=$EUB d=$DLB k=$K_SCENARIOS, MAP RAND) on GPU $GPU' && \
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
  --env.map-dir resources/drive/binaries/nuplan \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.k-scenarios $K_SCENARIOS \
  --env.conditioning.type none \
  --env.reward-lane-align $LANE_REWARD \
  --env.co-player-enabled 1 \
  --env.co-player-policy.policy-path $CKPT \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.collision-weight-lb $COLLISION_LB \
  --env.co-player-policy.conditioning.collision-weight-ub 0 \
  --env.co-player-policy.conditioning.offroad-weight-lb $OFFROAD_LB \
  --env.co-player-policy.conditioning.offroad-weight-ub 0 \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub $EUB \
  --env.co-player-policy.conditioning.discount-weight-lb $DLB \
  --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario True \
  --eval.map-dir resources/drive/binaries/nuplan \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS k=$K_SCENARIOS adaptive runs (MAP RAND) in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
