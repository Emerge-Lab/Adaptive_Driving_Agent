#!/bin/bash
set -e

# k=2 adaptive ego training with DIVERSE PARTNERS (per-episode sampling).
#
# Pairs the ego with one of the wide-conditioning partners trained in the
# entropy sweep (script: nuplan_transformer_local_201_entropy_sweep.sh).
# Partner conditioning is sampled once per EPISODE from the partner's wide
# training range (e.g. for the e=1.00 partner, entropy ∈ [0, 1.0]) and
# stays fixed within the episode. Different episodes get very different
# partner behaviors — the ego must identify partner type from s_0
# observations and apply that identification in s_1 (against the SAME
# partner) to drive well.
#
# Why no map_rand: map_rand causes ego batch row → agent identity
# misalignment, which corrupts the K/V cache content. With map_rand=False
# and per-episode partner sampling, agent identities stay stable across
# the boundary but partners differ across episodes — the ego's cache
# encodes partner type from s_0 and that encoding is still valid in s_1.
#
# Default GPUs: 0-4 (5 ego runs, one per partner).
#
# Override GPUs:    GPUS="0 1 2 3 4" bash <script>     (default already 0-4)
# Stop everything:  tmux kill-session -t adaptive_local_k2_201_diverse

# (wandb_id  entropy_ub  discount_lb)  -- 5 partners from the entropy sweep.
# The (entropy_lb, entropy_ub) range is what each partner was TRAINED with;
# at adaptive-ego training time we sample conditioning across the partner's
# trained range so the partner stays in-distribution.
COPLAYERS=(
  "miku2puk 0.05 0.4"
  "2e029h15 0.10 0.4"
  "m2ygolog 0.20 0.4"
  "6rauydj2 0.50 0.4"
  "n48teqjs 1.00 0.4"
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

# Same memory profile as the existing k=2/201 launchers (proven nw=32 nv=32).
NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"0 1 2 3 4"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#COPLAYERS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than coplayer slots (${#COPLAYERS[@]})" >&2
  exit 1
fi

SESSION=adaptive_local_k2_201_diverse
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r ID EUB DLB <<< "${COPLAYERS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}"
  TAG="adaptive_diverse_vs_${ID}_e${EUB}_d${DLB}_lane${LANE_REWARD}_k2_201_cr"
  CKPT="experiments/puffer_drive_${ID}.pt"

  if [ "$i" -ne 0 ]; then
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running exp${i}: diverse vs $ID (e_ub=$EUB d_lb=$DLB k=$K_SCENARIOS, condition_rand=True) on GPU $GPU' && \
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
  --env.map-rand-per-scenario False \
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS k=$K_SCENARIOS adaptive (DIVERSE) runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "Each ego paired with one entropy-sweep partner; partner conditioning resamples per scenario."
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
