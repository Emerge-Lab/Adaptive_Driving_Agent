#!/bin/bash
set -e

# Resume the 4 killed k=2 adaptive runs from their latest local checkpoints.
# Mirrors nuplan_transformer_local.sh (same conditioning per co-player) but
# loads each policy from disk via --load-model-path and uses a new wandb tag
# suffixed _resume so the new runs are identifiable in wandb.
#
# Runs out of /workspace/ADA-cp-gpu (the worktree). The k=2 baseline already
# saturates a 32 GiB GPU at nw=32 + torch.compile + minibatch_size=36400; adding
# the centralized co-player on the same GPU (--env.external-co-player-actions
# True) pushes past the GPU memory ceiling and OOMs during a train step. So we
# do NOT enable that flag here — the modest k=2 speedup (~1.36x) isn't worth
# the GPU memory pressure. cpu_offload also disabled (k=2 obs buffer fits on
# GPU; offloading would just add H2D overhead).
#
# Caveats:
#   - This creates NEW wandb runs (not a continuation of the killed ones).
#     Find them in adaptive_aligned with the matching `_resume` tag and pair
#     them visually with the original wid.
#   - Optimizer momentum, LR scheduler progress, and the global_step counter
#     all reset. Cosine LR annealing will start from peak again. If that
#     destabilizes a converged-ish policy, drop --train.learning-rate by ~3x.
#   - The latest model_*.pt for each wid is selected at launch time.
#   - The worktree's experiments/ is a symlink to /workspace/ADA/experiments,
#     so checkpoint paths resolve correctly.
#
# Override GPUs:    GPUS="0 1 2 3" bash <script>     (default already 0-3)
# Stop everything:  tmux kill-session -t adaptive_local_resume

# Each entry: ORIGINAL_WANDB_ID  COPLAYER_ID  ENTROPY_UB  DISCOUNT_LB
RUNS=(
  "1b4sk4o5 hntq6ykn 0.01 0.8"
  "qi3exrx1 ureqzfe9 0.01 0.6"
  "ldncmebn ampggcji 0.10 0.8"
  "koc6x4pp q1mtzo02 0.10 0.6"
)

# --- shared params (must match the original launcher) ---
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

GPUS=${GPUS:-"0 1 2 3"}
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
SESSION=adaptive_local_resume
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for i in "${!RUNS[@]}"; do
  read -r WID CPID EUB DLB <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  CKPT="${RESUME_CKPTS[$i]}"
  TAG="adaptive_vs_${CPID}_e${EUB}_d${DLB}_lane${LANE_REWARD}_resume_from_${WID}"
  COPLAYER_CKPT="experiments/puffer_drive_${CPID}.pt"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Resuming exp${i}: from $WID (co-player=$CPID, e=$EUB d=$DLB) on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --load-id $WID \
  --load-model-path $CKPT \
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
  --eval.map-dir resources/drive/binaries/nuplan \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Resumed ${#RUNS[@]} k=2 adaptive runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]:0:${#RUNS[@]}}"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
