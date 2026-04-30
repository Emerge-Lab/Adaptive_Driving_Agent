#!/bin/bash
set -e

# k_max=4 adaptive ego training with k_eff CURRICULUM. The ego's K/V
# cache is reset at within-episode scenario boundaries based on the
# stage:
#   Stage 0 (k_eff=1): reset at all 3 within-episode boundaries
#                      (effectively trains 4 independent scenarios)
#   Stage 1 (k_eff=2): reset only at the middle boundary s_1→s_2
#                      (two clean k=2 sub-episodes per episode)
#   Stage 2 (k_eff=4): no within-episode resets (full k=4 cache)
# Each stage = 30 episodes per worker.
#
# Reset mechanism: at boundaries to cut, drive.py sets
# truncations[ego_ids]=1 + terminals[ego_ids]=1. pufferl drops the
# transformer cache via done_mask=t+d during eval, and the training
# pass blocks cross-boundary attention via create_episode_mask.
# Both eval and train see the same effective context, so no
# rollout/training divergence.
#
# Comparison: 2 runs (curriculum vs no-curriculum), both at k_max=4
# horizon=804, partner = n48teqjs at full e_ub=1.0. The only varying
# factor is k_eff_curriculum_enabled.
#
# Memory: horizon=804 is ~2× the k=2 (horizon=402) runs, so per-run
# RAM ≈ 230GB. Two runs → ≈460GB peak. Cannot run alongside the k=2
# entropy curriculum ablation (which already uses ~470GB).
#
# Default GPUs: 4-5 (the entropy ablation occupies 0-3).
#
# Override GPUs:    GPUS="4 5" bash <script>
# Stop everything:  tmux kill-session -t adaptive_local_k4_201_curriculum

# (label, k_eff_curriculum_enabled)
RUNS=(
  "curr_k4    True"
  "nocurr_k4  False"
)

COPLAYER=experiments/puffer_drive_n48teqjs.pt

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_LB=0.4
DISCOUNT_UB=1
ENTROPY_UB=1.00
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=4
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 804

NUM_WORKERS=32
NUM_ENVS=32
# horizon doubled vs k=2 runs (804 vs 402); halve mb_mult to keep
# minibatch_size = 50 * 804 = 40200 (same as 100 * 402).
MINIBATCH_MULTIPLIER=50
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"4 5"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#RUNS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than runs (${#RUNS[@]})" >&2
  exit 1
fi

SESSION=adaptive_local_k4_201_curriculum
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r LABEL KCURR <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}_${LABEL//./}"
  TAG="adaptive_kcurr_${LABEL}_vs_n48teqjs_lane${LANE_REWARD}_k4_201"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t "$SESSION:exp0" "$WIN"
  else
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i} ${LABEL}: k_max=$K_SCENARIOS k_eff_curriculum=${KCURR} on GPU $GPU' && \
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
  --env.co-player-policy.policy-path $COPLAYER \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.transformer.horizon $SCENARIO_LENGTH \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.collision-weight-lb $COLLISION_LB \
  --env.co-player-policy.conditioning.collision-weight-ub 0 \
  --env.co-player-policy.conditioning.offroad-weight-lb $OFFROAD_LB \
  --env.co-player-policy.conditioning.offroad-weight-ub 0 \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub $ENTROPY_UB \
  --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
  --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --env.k-eff-curriculum-enabled $KCURR \
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS k_max=$K_SCENARIOS k_eff curriculum runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "  exp0 curr_k4    : k_eff annealed 1 → 2 → 4 over 3 stages"
echo "  exp1 nocurr_k4  : k_eff fixed at 4"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
