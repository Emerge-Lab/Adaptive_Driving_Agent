#!/bin/bash
set -e

# Resume the 4 killed k=3 adaptive runs from their latest local checkpoints.
# Mirrors nuplan_transformer_local_k3.sh but loads each policy via
# --load-model-path. New wandb runs created with `_resume` tag suffix.
#
# Runs out of /workspace/ADA-cp-gpu and enables BOTH new optimizations:
#   --env.external-co-player-actions True : centralized GPU co-player forward
#                                           (the big k=3 win — see equivalence
#                                           test in /tmp/morning_summary.md)
#   --train.cpu-offload True              : moves the obs buffer to pinned RAM
#                                           so we can use nw=32 without GPU OOM
#                                           (was OOM-bound at nw=20 on 32 GiB)
# Combined: ~22x speedup over the original nw=16/no-flags baseline (110h ETA
# becomes ~5h on a single GPU).
#
# RAM CAVEAT: cpu_offload puts ~120 GiB obs buffer in pinned RAM. With 4
# concurrent k=3 runs on the box, that's 480 GiB of 503 total — borderline.
# If you launch other large jobs on the box at the same time, the kernel may
# OOM-kill (this is what happened when nw=48 was tried). Monitor `free -g`.
#
# Caveats: same resume notes as nuplan_transformer_local_resume.sh. Optimizer/
# LR state resets; cosine annealing restarts. If destabilizes, lower
# --train.learning-rate.
#
# Override GPUs:    GPUS="4 5 6 7" bash <script>     (default already 4-7)
# Stop everything:  tmux kill-session -t adaptive_local_k3_resume

# Each entry: ORIGINAL_WANDB_ID  COPLAYER_ID  ENTROPY_UB  DISCOUNT_LB
RUNS=(
  "y9tges7d ampggcji 0.10 0.8"
  "5p6tl3pt q1mtzo02 0.10 0.6"
  "3p5ome2t ureqzfe9 0.01 0.6"
  "bzzosxsg hntq6ykn 0.01 0.8"
)

# --- shared params (must match the original launcher) ---
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=3
SCENARIO_LENGTH=91
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 273

# With cpu_offload=True the obs buffer lives in pinned RAM, freeing the GPU
# from the 32 GiB ceiling that previously forced nw=16 here. nw=32 fits easily
# on a 5090 (peak ~18 GiB) and matches the adaptive ini default.
NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=200
MAX_MINIBATCH_SIZE=54600

GPUS=${GPUS:-"4 5 6 7"}
read -r -a GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -lt "${#RUNS[@]}" ]; then
  echo "ERROR: need ${#RUNS[@]} GPUs, got ${#GPU_ARR[@]} ($GPUS)" >&2
  exit 1
fi

# --- step 1: snapshot latest co-player checkpoint to top-level .pt ---
echo "Snapshotting latest co-player checkpoints…"
for entry in "${RUNS[@]}"; do
  read -r WID CPID _ _ <<< "$entry"
  src=$(ls /workspace/ADA/experiments/puffer_drive_$CPID/model_puffer_drive_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$src" ]; then
    echo "  ERROR: no co-player checkpoint for $CPID" >&2
    exit 1
  fi
  cp -f "$src" "/workspace/ADA/experiments/puffer_drive_$CPID.pt"
  echo "  $CPID: $(basename "$src")"
done

# --- step 2: locate latest adaptive checkpoint for each killed run ---
echo "Locating resume checkpoints…"
declare -a RESUME_CKPTS
for entry in "${RUNS[@]}"; do
  read -r WID _ _ _ <<< "$entry"
  ckpt=$(ls /workspace/ADA/experiments/puffer_adaptive_drive_$WID/model_puffer_adaptive_drive_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$ckpt" ]; then
    echo "  ERROR: no adaptive checkpoint for $WID" >&2
    exit 1
  fi
  RESUME_CKPTS+=("$ckpt")
  echo "  $WID: $(basename "$ckpt")"
done

# --- step 3: launch in tmux ---
SESSION=adaptive_local_k3_resume
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for i in "${!RUNS[@]}"; do
  read -r WID CPID EUB DLB <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  CKPT="${RESUME_CKPTS[$i]}"
  TAG="adaptive_vs_${CPID}_e${EUB}_d${DLB}_lane${LANE_REWARD}_k${K_SCENARIOS}_resume_from_${WID}"
  COPLAYER_CKPT="experiments/puffer_drive_${CPID}.pt"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Resuming exp${i}: from $WID (co-player=$CPID, e=$EUB d=$DLB k=$K_SCENARIOS) on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --load-id $WID \
  --load-model-path $CKPT \
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
  --env.co-player-policy.policy-path $COPLAYER_CKPT \
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
  --eval.map-dir resources/drive/binaries/nuplan \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Resumed ${#RUNS[@]} k=$K_SCENARIOS adaptive runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]:0:${#RUNS[@]}}"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
