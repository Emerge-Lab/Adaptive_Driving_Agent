#!/bin/bash
set -e

# Local 4-GPU launcher (GPUs 4-7) for additional NuPlan Transformer co-players.
# Mirrors the tmux-window-per-experiment pattern of human_align_local.sh.
#
# Grid: e ∈ {0.01, 0.1} × d ∈ {0.8, 0.6}
# Fixed: collision_weight_lb=-2, offroad_weight_lb=-2, lane_reward=0.01

ENTROPY_UB=(0.01 0.10 0.01 0.10)
DISCOUNT_LB=(0.8  0.8  0.6  0.6)

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=5000
SEED=42

SESSION=coplayer_nuplan_tfm_local
tmux new-session -d -s "$SESSION" -n exp0

for i in "${!ENTROPY_UB[@]}"; do
  EUB=${ENTROPY_UB[$i]}
  DLB=${DISCOUNT_LB[$i]}
  GPU=$((i+4))
  WIN="exp${i}"
  TAG="coplayer_nuplan_transformer_local_e${EUB}_d${DLB}_c${COLLISION_LB}_o${OFFROAD_LB}_lane${LANE_REWARD}"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: entropy_ub=$EUB discount_lb=$DLB on GPU $GPU' && \
xvfb-run -a puffer train puffer_drive \
  --wandb --wandb-project ada_new_coplayers \
  --tag $TAG \
  --env.map-dir resources/drive/binaries/nuplan \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.reward-lane-align $LANE_REWARD \
  --env.conditioning.type all \
  --env.conditioning.collision-weight-lb $COLLISION_LB \
  --env.conditioning.collision-weight-ub 0 \
  --env.conditioning.offroad-weight-lb $OFFROAD_LB \
  --env.conditioning.offroad-weight-ub 0 \
  --env.conditioning.entropy-weight-lb 0 \
  --env.conditioning.entropy-weight-ub $EUB \
  --env.conditioning.discount-weight-lb $DLB \
  --env.conditioning.discount-weight-ub $DISCOUNT_UB \
  --policy-architecture Transformer \
  --train.context-length 91 \
  --train.horizon 91 \
  --train.learning-rate 0.003 \
  --train.checkpoint-interval 50 \
  --train.seed $SEED \
  --eval.map-dir resources/drive/binaries/nuplan"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo "Launched ${#ENTROPY_UB[@]} runs in tmux session '$SESSION'."
echo "Attach: tmux attach -t $SESSION"
echo "Switch windows: Ctrl-b 0..3  (or Ctrl-b n / Ctrl-b p)"
echo "Detach: Ctrl-b d"
echo "Stop all: tmux kill-session -t $SESSION"
