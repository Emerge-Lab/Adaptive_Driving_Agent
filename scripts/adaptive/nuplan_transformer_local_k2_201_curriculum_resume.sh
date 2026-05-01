#!/bin/bash
set -e

# Resume the 3 killed runs from the entropy_ub curriculum ablation.
# exp0 (curr_e1.0, wandb=uufybjgm) finished cleanly and is NOT resumed.
#
# Resume targets (load-id is original wandb run; episode counts inferred
# from `--train.checkpoint-interval 10` so the latest ckpt is the highest
# multiple of 10 below the death epoch):
#   exp1 nocurr_e1.0  → 7mnrbviz (died ep 65, latest ckpt ep 60). curr=False
#   exp2 curr_e0.5    → hprfn8dc (died ep 85, latest ckpt ep 80). curr=True;
#                        --env.entropy-curriculum-episodes-start 80 to skip
#                        stages 0,1 (would otherwise reset to stage 0).
#   exp3 nocurr_e0.5  → 7wm1sk5v (died ep 135, latest ckpt ep 130). curr=False
#
# Caveats:
#   - NEW wandb run (not continuation). Pair visually with original WID.
#   - Optimizer momentum, global_step, and epoch ARE restored from the
#     sibling trainer_state.pt (pufferl.py:280) — cosine LR resumes
#     mid-schedule, not at peak. So we keep the default LR.
#   - Curriculum: only exp2 needs the start-episode knob (drive.py kwarg
#     entropy_curriculum_episodes_start). For exp1 / exp3 (nocurr) it's a
#     no-op.
#
# Default GPUs: 4 5 6. GPUs 1-3 hold stuck CUDA contexts from the dead
# original runs (driver doesn't release without a container restart);
# GPUs 4-7 are clean. Override if you have GPUs 1-3 cleared.
#
# Override GPUs:    GPUS="4 5 6" bash <script>
# Stop everything:  tmux kill-session -t adaptive_local_k2_201_curriculum_resume

# (label, final_eub, curriculum_enabled, coplayer_id, original_wid,
#  curriculum_episodes_start)
RUNS=(
  "nocurr_e1.0  1.00 False n48teqjs 7mnrbviz 0"
  "curr_e0.5    0.50 True  6rauydj2 hprfn8dc 80"
  "nocurr_e0.5  0.50 False 6rauydj2 7wm1sk5v 0"
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

NUM_WORKERS=24
NUM_ENVS=24
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"4 5 6"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#RUNS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than runs (${#RUNS[@]})" >&2
  exit 1
fi

# Locate latest adaptive checkpoint per WID.
echo "Locating resume checkpoints…"
declare -a RESUME_CKPTS
for entry in "${RUNS[@]}"; do
  read -r _ _ _ _ WID _ <<< "$entry"
  ckpt=$(ls /workspace/ADA/experiments/puffer_adaptive_drive_$WID/model_puffer_adaptive_drive_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$ckpt" ]; then
    echo "  ERROR: no checkpoint for $WID" >&2
    exit 1
  fi
  RESUME_CKPTS+=("$ckpt")
  echo "  $WID: $(basename "$ckpt")"
done

SESSION=adaptive_local_k2_201_curriculum_resume
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r LABEL EUB CURR PARTNER_ID WID EPISODES_START <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}_${LABEL//./}"
  TAG="adaptive_diverse_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201_resume_from_${WID}"
  COPLAYER="experiments/puffer_drive_${PARTNER_ID}.pt"
  CKPT="${RESUME_CKPTS[$i]}"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t "$SESSION:exp0" "$WIN"
  else
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Resuming exp${i} ${LABEL}: from $WID (eub=${EUB} curr=${CURR} ep_start=${EPISODES_START}) on GPU $GPU' && \
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
  --train.render-interval 30 \
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
  --env.entropy-curriculum-episodes-start $EPISODES_START \
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.human-replay-num-rollouts 50 \
  --eval.eval-interval 5"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Resumed $N_RUNS curriculum-ablation runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "  exp0 nocurr_e1.0 (resume from epoch 60)  : entropy_ub fixed at 1.0"
echo "  exp1 curr_e0.5   (resume from epoch 80)  : entropy_ub annealed 0.025 → 0.5; starts in stage 2"
echo "  exp2 nocurr_e0.5 (resume from epoch 130) : entropy_ub fixed at 0.5"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
