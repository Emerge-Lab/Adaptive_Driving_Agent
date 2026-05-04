#!/bin/bash
set -e

# k-scenarios sweep against the 2 deterministic partners (p005, p010).
# γ=0.995, lane=0.025, gb=2 (from adaptive.ini), no curriculum.
# Direct comparator to the k=2 vanilla baselines currently training.
#
# Default: 4 runs (k ∈ {3, 4} × partner ∈ {miku2puk, 2e029h15}) on GPUs 0-3.
# Optional 5th: a seed=2 of k=4_2e029h15 on GPU 4 for variance check.
#
# Override: GPUS="0 1 2 3 4" RUN_INDICES="0 1 2 3 4" bash <script>
# Stop:     tmux kill-session -t adaptive_k_sweep

# (label, k, partner_id, entropy_ub, seed)
RUNS=(
  "k2_p010_s42  2  2e029h15  0.10  42"
  "k2_p010_s55  2  2e029h15  0.10  55"
  "k2_p010_s77  2  2e029h15  0.10  77"
  "k4_p010_s42  4  2e029h15  0.10  42"
  "k4_p010_s55  4  2e029h15  0.10  55"
  "k4_p010_s77  4  2e029h15  0.10  77"
)

GAMMA=0.995
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.025
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SCENARIO_LENGTH=201

# Per-run pinned RAM scales as num_workers * num_envs * horizon (cpu_offload=True).
# With horizon=603 (k=3) or 804 (k=4), 32/32 would push 4 simultaneous runs
# past system RAM (current ≈800 GiB usable). Drop to 24/24 → ~56% of 32/32 mem.
# k-aware minibatch: keep effective batch ≈40k by scaling multiplier as horizon
# grows (must be divisible by horizon).
# 24/24 per user request. Memory budget for 6 trainings (3 k=2 + 3 k=4):
# 3 × 50 + 3 × 99 ≈ 447 GiB. Fits in 1007 GiB total.
NUM_WORKERS=${K_SWEEP_NUM_WORKERS:-24}
NUM_ENVS=${K_SWEEP_NUM_ENVS:-24}
MAX_MINIBATCH_SIZE=80400

RUN_INDICES=${RUN_INDICES:-"0 1 2 3"}
GPUS=${GPUS:-"0 1 2 3"}
read -r -a IDX_ARR <<< "$RUN_INDICES"
read -r -a GPU_ARR <<< "$GPUS"
if [ "${#IDX_ARR[@]}" -ne "${#GPU_ARR[@]}" ]; then
  echo "ERROR: RUN_INDICES count (${#IDX_ARR[@]}) != GPUS count (${#GPU_ARR[@]})" >&2
  exit 1
fi

SESSION=${SESSION:-adaptive_k_sweep}
tmux has-session -t "$SESSION" 2>/dev/null && echo "Session $SESSION exists; appending windows" || tmux new-session -d -s "$SESSION" -n placeholder

for ((i=0; i<${#IDX_ARR[@]}; i++)); do
  IDX=${IDX_ARR[$i]}
  GPU=${GPU_ARR[$i]}
  read -r LABEL K PARTNER_ID EUB SEED <<< "${RUNS[$IDX]}"
  HORIZON=$((K * SCENARIO_LENGTH))
  # k-aware multiplier: k=2→100, k=3→67, k=4→50 → effective minibatch ≈40k each
  case $K in
    2) MINIBATCH_MULTIPLIER=100 ;;
    3) MINIBATCH_MULTIPLIER=67  ;;  # 67 * 603 = 40401
    4) MINIBATCH_MULTIPLIER=50  ;;  # 50 * 804 = 40200
    *) MINIBATCH_MULTIPLIER=$(( 40200 / HORIZON )) ;;
  esac
  WIN="${LABEL}"
  TAG="adaptive_${LABEL}_g${GAMMA}_lane${LANE_REWARD}_gb2"
  COPLAYER="experiments/puffer_drive_${PARTNER_ID}.pt"

  tmux new-window -t "$SESSION" -n "$WIN"

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo '${LABEL}: k=$K partner=$PARTNER_ID e_ub=$EUB seed=$SEED gpu=$GPU horizon=$HORIZON' && \
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
  --env.k-scenarios $K \
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
  --eval.human-replay-num-rollouts 10 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540 \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

tmux kill-window -t "$SESSION:placeholder" 2>/dev/null || true

echo
echo "Launched ${#IDX_ARR[@]} k-sweep runs in tmux session '$SESSION'"
for ((i=0; i<${#IDX_ARR[@]}; i++)); do
  IDX=${IDX_ARR[$i]}
  read -r LABEL K PARTNER_ID EUB SEED <<< "${RUNS[$IDX]}"
  echo "  GPU ${GPU_ARR[$i]}: $LABEL  (k=$K partner=$PARTNER_ID seed=$SEED)"
done
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
