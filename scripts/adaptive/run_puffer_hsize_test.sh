#!/usr/bin/env bash
# Args: HIDDEN_SIZE FIX_FASTPATH(0|1) LOG_PATH TIMEOUT_SEC
set -u
HIDDEN_SIZE="$1"
FIX_FASTPATH="$2"
LOG_PATH="$3"
TIMEOUT_SEC="${4:-180}"

# Fixed config — matches cluster_hiddensize_ablation.sh exactly (the recipe that failed).
PARTNER_ID=2e029h15
ENTROPY_UB=0.10
K_SCENARIOS=4
LR=3e-3
ENT_COEF=0.005
COPLAYER_PATH="experiments/puffer_drive_${PARTNER_ID}.pt"
GAMMA=0.995
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.05
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))
COLLISION_PENALTY_EGO=-0.5
OFFROAD_PENALTY_EGO=-0.5
NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=50
MAX_MINIBATCH_SIZE=$((50 * HORIZON))
TOTAL_TIMESTEPS=50000000  # batch_size ≈ num_ego*ctx*nworkers (~13M); need >>batch for train_epochs > 0

if [[ "$FIX_FASTPATH" == "1" ]]; then
  CMD="xvfb-run -a python scripts/adaptive/puffer_no_mha_fastpath.py"
  LABEL="fix"
else
  CMD="xvfb-run -a puffer"
  LABEL="plain"
fi

echo "[hsize_test] hidden=$HIDDEN_SIZE  variant=$LABEL  log=$LOG_PATH" | tee -a "$LOG_PATH"

# Run inside singularity; capture full stdout+stderr.
timeout --kill-after=10s "${TIMEOUT_SEC}s" \
singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate
   export WANDB_MODE=disabled
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export PUFFER_TRANSFORMER_LEGACY_EVAL=1
   $CMD train puffer_adaptive_drive \
     --policy-architecture Transformer --rnn-name Transformer \
     --policy.hidden-size $HIDDEN_SIZE \
     --transformer.input-size $HIDDEN_SIZE \
     --transformer.hidden-size $HIDDEN_SIZE \
     --train.gamma $GAMMA \
     --train.learning-rate $LR \
     --train.ent-coef $ENT_COEF \
     --train.horizon $HORIZON \
     --train.minibatch-multiplier $MINIBATCH_MULTIPLIER \
     --train.max-minibatch-size $MAX_MINIBATCH_SIZE \
     --train.cpu-offload True \
     --train.checkpoint-interval 100000 \
     --train.render-interval 100000 \
     --train.seed 42 \
     --train.total-timesteps $TOTAL_TIMESTEPS \
     --vec.num-workers $NUM_WORKERS --vec.num-envs $NUM_ENVS --vec.batch-size 32 \
     --env.map-dir resources/drive/binaries/nuplan_201 \
     --env.num-maps $NUPLAN_NUM_MAPS \
     --env.scenario-length $SCENARIO_LENGTH \
     --env.k-scenarios $K_SCENARIOS \
     --env.goal-behavior 3 \
     --env.conditioning.type none \
     --env.reward-lane-align $LANE_REWARD \
     --env.reward-vehicle-collision $COLLISION_PENALTY_EGO \
     --env.reward-offroad-collision $OFFROAD_PENALTY_EGO \
     --env.co-player-enabled 1 \
     --env.co-player-policy.policy-path $COPLAYER_PATH \
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
     --env.entropy-curriculum-enabled False
 " >> "$LOG_PATH" 2>&1
EC=$?
echo "[hsize_test] exit_code=$EC  hidden=$HIDDEN_SIZE  variant=$LABEL" | tee -a "$LOG_PATH"
exit $EC
