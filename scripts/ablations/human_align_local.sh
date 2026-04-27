#!/bin/bash
set -e

# Local 8-GPU launcher for human-align ablation.
# One tmux window per experiment, each pinned to a single GPU.

COLLISION_WEIGHTS=(-3 -3 -2 -2)
OFFROAD_WEIGHTS=(-2.0 -0.5 -2.0 -0.5)

DISCOUNT_UB=0.995
SEED=42
NUPLAN_NUM_MAPS=4999

SESSION=human_align_redo
tmux new-session -d -s "$SESSION" -n exp0

for i in "${!COLLISION_WEIGHTS[@]}"; do
  C=${COLLISION_WEIGHTS[$i]}
  O=${OFFROAD_WEIGHTS[$i]}
  L=${REWARD_LANE[$i]}
  WIN="exp${i}"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$((i+4)) && \
echo 'Running exp${i}: collision_weight_lb=$C, offroad_weight_lb=$O on GPU $i' && \
xvfb-run -a puffer train puffer_drive \
  --wandb --wandb-project human-align-ablation \
  --tag human_ablation_lane_rewards_apr26_redo \
  --env.map-dir resources/drive/binaries/nuplan \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.conditioning.type all \
  --env.conditioning.collision-weight-lb $C \
  --env.conditioning.collision-weight-ub 0 \
  --env.conditioning.offroad-weight-lb $O \
  --env.conditioning.offroad-weight-ub 0 \
  --env.conditioning.discount-weight-lb 0.8 \
  --env.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.reward-lane-align 0.01 \
  --env.reward-vel-align 1.0 \
  --policy-architecture Transformer \
  --train.context-length 91 \
  --train.horizon 91 \
  --train.seed $SEED \
  --eval.map-dir resources/drive/binaries/nuplan"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo "Launched ${#COLLISION_WEIGHTS[@]} runs in tmux session '$SESSION'."
echo "Attach: tmux attach -t $SESSION"
echo "Switch windows: Ctrl-b 0..7  (or Ctrl-b n / Ctrl-b p)"
echo "Detach: Ctrl-b d"
