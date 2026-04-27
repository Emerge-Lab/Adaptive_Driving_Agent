#!/bin/bash
set -e

# Same as nuplan_transformer_local.sh but with k_scenarios=3 (horizon=273).
# Defaults to GPUs 4-7. Reduces num_workers and minibatch settings so the
# 1.5x larger obs buffer fits on a 32 GiB 5090.
#
# Override GPUs:    GPUS="4 5 6 7" bash <script>     (default already 4-7)
# Stop everything:  tmux kill-session -t adaptive_local_k3

# --- co-player checkpoints (same 4 as the k=2 launcher) ---
COPLAYERS=(
  "ampggcji 0.10 0.8"
  "q1mtzo02 0.10 0.6"
  "ureqzfe9 0.01 0.6"
  "hntq6ykn 0.01 0.8"
)

# --- shared params ---
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=3
SCENARIO_LENGTH=91
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 273

# Memory-fit knobs for k=3 on a 32 GiB GPU:
#   nw=16  -> obs buffer ~16.6 GiB (nw=20 OOMs at the first train step;
#            the train-time peak combined with the obs buffer leaves no slack
#            on a 32 GiB 5090)
#   MM=200, MAXMB=54600 -> minibatch=200*273=54600 (no grad accum, segment-aligned)
NUM_WORKERS=16
NUM_ENVS=16
MINIBATCH_MULTIPLIER=200
MAX_MINIBATCH_SIZE=54600

GPUS=${GPUS:-"4 5 6 7"}
read -r -a GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -lt "${#COPLAYERS[@]}" ]; then
  echo "ERROR: need ${#COPLAYERS[@]} GPUs, got ${#GPU_ARR[@]} ($GPUS)" >&2
  exit 1
fi

# --- step 1: snapshot latest co-player checkpoint to top-level .pt ---
echo "Snapshotting latest checkpoints…"
for entry in "${COPLAYERS[@]}"; do
  read -r ID _ _ <<< "$entry"
  src=$(ls /workspace/ADA/experiments/puffer_drive_$ID/model_puffer_drive_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$src" ]; then
    echo "  ERROR: no checkpoint found for puffer_drive_$ID" >&2
    exit 1
  fi
  dst=/workspace/ADA/experiments/puffer_drive_$ID.pt
  cp -f "$src" "$dst"
  echo "  $ID: $(basename "$src")  →  $(basename "$dst")"
done

# --- step 2: launch in tmux ---
SESSION=adaptive_local_k3
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for i in "${!COPLAYERS[@]}"; do
  read -r ID EUB DLB <<< "${COPLAYERS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="adaptive_vs_${ID}_e${EUB}_d${DLB}_lane${LANE_REWARD}_k${K_SCENARIOS}"
  CKPT="experiments/puffer_drive_${ID}.pt"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: co-player=$ID (e=$EUB d=$DLB k=$K_SCENARIOS) on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.horizon $HORIZON \
  --train.minibatch-multiplier $MINIBATCH_MULTIPLIER \
  --train.max-minibatch-size $MAX_MINIBATCH_SIZE \
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
  --eval.map-dir resources/drive/binaries/nuplan \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched ${#COPLAYERS[@]} adaptive (k=$K_SCENARIOS) runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]:0:${#COPLAYERS[@]}}"
echo "Attach: tmux attach -t $SESSION"
echo "Switch windows: Ctrl-b 0..3"
echo "Detach: Ctrl-b d"
echo "Stop all: tmux kill-session -t $SESSION"
