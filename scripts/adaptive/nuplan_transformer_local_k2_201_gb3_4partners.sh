#!/bin/bash
set -e

# 4 k=2 transformer runs, gb=3 (GOAL_TRIAL), one per partner.
# Train on nuplan_201 (5000 maps), eval + renders on nuplan_hard (540 maps).
# Renders use eval.map-dir via the render_videos patch in pufferlib/utils.py
# (eval_map_dir override added 2026-05-15).
# Standard config from k2_201_partner_sweep.sh (γ=0.995, lane=0.025) + 2B steps.
#
# Default GPUs: 0-3 (all 4 free).
# Override GPUs:    GPUS="0 1 2 3" bash <script>
# Stop everything:  tmux kill-session -t ada_k2_gb3_4partners

RUNS=(
  "p005_miku2puk_gb3  miku2puk  0.05"
  "p010_2e029h15_gb3  2e029h15  0.10"
  "p020_m2ygolog_gb3  m2ygolog  0.20"
  "p050_6rauydj2_gb3  6rauydj2  0.50"
)

GAMMA=0.995
COLLISION_LB=-2; OFFROAD_LB=-2
LANE_REWARD=0.025
DISCOUNT_LB=0.4; DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42
TOTAL_TIMESTEPS=2000000000   # 2B

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))   # 402

NUM_WORKERS=8; NUM_ENVS=8     # container memory.max=132GiB; per-run pinned ≈ nw*2.83GiB; 4 runs fit at nw≤9
MINIBATCH_MULTIPLIER=100; MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"0 1 2 3"}
read -r -a GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -ne "${#RUNS[@]}" ]; then
  echo "ERROR: GPUs count (${#GPU_ARR[@]}) != runs count (${#RUNS[@]})" >&2
  exit 1
fi

SESSION=${SESSION:-ada_k2_gb3_4partners}
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n placeholder

for ((i=0; i<${#RUNS[@]}; i++)); do
  read -r LABEL PARTNER_ID EUB <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="$LABEL"
  TAG="ada_k2_gb3_${LABEL}_g${GAMMA}_lane${LANE_REWARD}"
  COPLAYER="experiments/puffer_drive_${PARTNER_ID}.pt"

  tmux new-window -t "$SESSION" -n "$WIN"

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
export WANDB_MODE=online && \
echo 'Running $LABEL: partner=$PARTNER_ID e_ub=$EUB gamma=$GAMMA gb=3 on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer --rnn-name Transformer \
  --train.gamma $GAMMA \
  --train.horizon $HORIZON \
  --train.minibatch-multiplier $MINIBATCH_MULTIPLIER \
  --train.max-minibatch-size $MAX_MINIBATCH_SIZE \
  --train.cpu-offload True \
  --train.checkpoint-interval 10 \
  --train.render-interval 30 \
  --train.seed $SEED \
  --train.total-timesteps $TOTAL_TIMESTEPS \
  --vec.num-workers $NUM_WORKERS --vec.num-envs $NUM_ENVS \
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.scenario-length $SCENARIO_LENGTH \
  --env.k-scenarios $K_SCENARIOS \
  --env.goal-behavior 3 \
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
  --eval.human-replay-num-rollouts 200 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540 \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

tmux kill-window -t "$SESSION:placeholder" 2>/dev/null || true

echo
echo "Launched ${#RUNS[@]} k=2 gb=3 runs in tmux session '$SESSION'"
for ((i=0; i<${#RUNS[@]}; i++)); do
  read -r LABEL PARTNER_ID EUB <<< "${RUNS[$i]}"
  echo "  GPU ${GPU_ARR[$i]}: $LABEL  (partner=$PARTNER_ID  e_ub=$EUB)"
done
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
