#!/bin/bash
set -e

# Local 4-GPU adaptive launcher (tmux pattern, mirrors human_align_local.sh).
# Trains 4 adaptive ego agents, one against each of the new local co-player
# checkpoints (collision_lb=-2, offroad_lb=-2, lane_reward=0.01).
#
# Each ego run uses a co-player conditioning band that matches what THAT
# co-player saw during its own training (so the policy is queried inside the
# distribution it learned).
#
# DEFAULT GPU ASSIGNMENT: 0,1,2,3. Override with: GPUS="4 5 6 7" bash <script>
# (Make sure the chosen GPUs are free — kill any prior adaptive_vs_* runs.)

# --- co-player checkpoints ---
# Each entry: WANDB_ID  ENTROPY_UB  DISCOUNT_LB
COPLAYERS=(
  "ampggcji 0.10 0.8"
  "q1mtzo02 0.10 0.6"
  "ureqzfe9 0.01 0.6"
  "hntq6ykn 0.01 0.8"
)

# --- shared adaptive-run params ---
COLLISION_LB=-2          # match co-player training
OFFROAD_LB=-2            # match co-player training
LANE_REWARD=0.01         # ego lane reward
DISCOUNT_UB=1            # match co-player training
NUPLAN_NUM_MAPS=4999
SEED=42

GPUS=${GPUS:-"0 1 2 3"}
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
SESSION=adaptive_local
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for i in "${!COPLAYERS[@]}"; do
  read -r ID EUB DLB <<< "${COPLAYERS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="adaptive_vs_${ID}_e${EUB}_d${DLB}_lane${LANE_REWARD}"
  CKPT="experiments/puffer_drive_${ID}.pt"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: co-player=$ID (e=$EUB d=$DLB) on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.horizon 182 \
  --train.checkpoint-interval 10 \
  --train.render-interval 10 \
  --train.seed $SEED \
  --env.map-dir resources/drive/binaries/nuplan \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.k-scenarios 2 \
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
echo "Launched ${#COPLAYERS[@]} adaptive runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]:0:${#COPLAYERS[@]}}"
echo "Attach: tmux attach -t $SESSION"
echo "Switch windows: Ctrl-b 0..3"
echo "Detach: Ctrl-b d"
echo "Stop all: tmux kill-session -t $SESSION"
