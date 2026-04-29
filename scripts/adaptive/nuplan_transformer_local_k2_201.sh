#!/bin/bash
set -e

# k=2 adaptive runs against the new 201-frame nuplan coplayers, with
# --env.map-rand-per-scenario True so scenario 1 is a fresh scene and the
# only signal that crosses the boundary is the ego policy's K/V cache.
# Yields a clean single ada_delta = score(s1) - score(s0) per episode.
#
# The coplayers (ocd1syvg, 60knfipr, tut3cuh1, xwthz2lg) were trained on
# resources/drive/binaries/nuplan_201 with --env.scenario-length 201, so
# this launcher matches both knobs. Each coplayer is paired with the same
# (entropy_ub, discount_lb) it trained with so the conditioning the ego
# samples actually exercises behaviors the partner has seen.
#
# Defaults to GPUs 0-3 (k=3 maprand is already using 4-7).
# Memory: cpu_offload True puts obs buffer in pinned RAM (frees GPU). With
# horizon=402 the obs buffer is 1.5x bigger than k=3's, so we drop num_workers
# to 16 to keep RAM usage comfortable alongside the live k=3 runs.
#
# Override GPUs:    GPUS="0 1 2 3" bash <script>     (default already 0-3)
# Stop everything:  tmux kill-session -t adaptive_local_k2_201

# (wandb_id, entropy_weight_ub, discount_weight_lb) - matches each
# coplayer's training conditioning so we sample inside its support.
COPLAYERS=(
  "ocd1syvg 0.01 0.8"
  "60knfipr 0.10 0.8"
  "tut3cuh1 0.01 0.6"
  "xwthz2lg 0.10 0.6"
)

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 402

# nw=32 nv=32 with cpu_offload puts obs in pinned RAM; per-run pinned
# scales as horizon (402) so each run is ~88 GiB. 4 runs ~352 GiB —
# fits the 382 GiB headroom but tight; only safe when no other adaptive
# runs are live. multiplier=100 -> minibatch=40200 (divisible by horizon).
NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#COPLAYERS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than coplayer slots (${#COPLAYERS[@]})" >&2
  exit 1
fi

# --- snapshot latest co-player checkpoints ---
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

SESSION=adaptive_local_k2_201
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r ID EUB DLB <<< "${COPLAYERS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="adaptive_vs_${ID}_e${EUB}_d${DLB}_lane${LANE_REWARD}_k2_201_mr"
  CKPT="experiments/puffer_drive_${ID}.pt"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: co-player=$ID (e=$EUB d=$DLB k=$K_SCENARIOS, 201-frame) on GPU $GPU' && \
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
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.scenario-length $SCENARIO_LENGTH \
  --env.k-scenarios $K_SCENARIOS \
  --env.conditioning.type none \
  --env.reward-lane-align $LANE_REWARD \
  --env.co-player-enabled 1 \
  --env.co-player-policy.policy-path $CKPT \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.transformer.horizon $SCENARIO_LENGTH \
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
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS k=$K_SCENARIOS adaptive runs (201-frame coplayers) in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
