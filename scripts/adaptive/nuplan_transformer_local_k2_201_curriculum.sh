#!/bin/bash
set -e

# k=2 adaptive ego training — 2x2 grid: {curriculum, no curriculum} ×
# {final entropy_ub = 0.5, 1.0}. Within each entropy_ub level, both runs
# share the SAME partner (matched to that level's trained range), so the
# only difference within a pair is curriculum on/off:
#   e=1.0 pair: vs n48teqjs (entropy-sweep partner trained ∈ [0, 1.0])
#   e=0.5 pair: vs 6rauydj2 (entropy-sweep partner trained ∈ [0, 0.5])
# (e=0.2 pair was dropped — host can fit 4 simultaneous runs at nw=32
# nv=32; per-process overhead dominates, so 6 doesn't fit even with
# shrunk per-run sizing.)
#
# Curriculum schedule (when enabled): 4 stages, each 30 episodes per worker
# (≈30 epochs at our nw=32 nv=32 setup). Stage k uses entropy_ub =
# stage_ratio[k] * final_ub, where ratios = [0.05, 0.20, 0.50, 1.00].
# All other conditioning dims (collision/offroad/discount) sample at full
# range throughout; only entropy_ub is annealed.
#
# Hypothesis: ada_delta_score peaks early then drifts toward 0 as scores
# saturate (we observed this in city-adapt). Curriculum should keep the
# task in an informative-difficulty regime for longer, surfacing more
# adaptation signal late in training.
#
# Default GPUs: 0-3 (one per run).
#
# Override GPUs:    GPUS="0 1 2 3" bash <script>
# Stop everything:  tmux kill-session -t adaptive_local_k2_201_curriculum

# (label, final entropy_ub, curriculum_enabled, coplayer_id)
RUNS=(
  "curr_e1.0    1.00 True  n48teqjs"
  "nocurr_e1.0  1.00 False n48teqjs"
  "curr_e0.5    0.50 True  6rauydj2"
  "nocurr_e0.5  0.50 False 6rauydj2"
)

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 402

NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#RUNS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than runs (${#RUNS[@]})" >&2
  exit 1
fi

SESSION=adaptive_local_k2_201_curriculum
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r LABEL EUB CURR PARTNER_ID <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  # Strip dots from LABEL so tmux doesn't read them as pane separators.
  WIN="exp${i}_${LABEL//./}"
  TAG="adaptive_diverse_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"
  COPLAYER="experiments/puffer_drive_${PARTNER_ID}.pt"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t "$SESSION:exp0" "$WIN"
  else
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i} ${LABEL}: final entropy_ub=${EUB} curriculum=${CURR} on GPU $GPU' && \
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
  --env.co-player-policy.conditioning.entropy-weight-ub $EUB \
  --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
  --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --env.entropy-curriculum-enabled $CURR \
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS curriculum-ablation runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "  exp0 curr_e1.0    vs n48teqjs : entropy_ub annealed 0.05 → 1.0 over 4 stages"
echo "  exp1 nocurr_e1.0  vs n48teqjs : entropy_ub fixed at 1.0"
echo "  exp2 curr_e0.5    vs 6rauydj2 : entropy_ub annealed 0.025 → 0.5 over 4 stages"
echo "  exp3 nocurr_e0.5  vs 6rauydj2 : entropy_ub fixed at 0.5"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
