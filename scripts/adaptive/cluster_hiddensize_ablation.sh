#!/bin/bash
#SBATCH --job-name=hsize_abl
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-11
#
# Hidden-size ablation for ego model.
# Fixed:  k=4, partner=2e029h15 (entropy_ub=0.10), all other hparams identical
# to cluster_coplayer_grid_k234.sh (the recipe used for the headline k=4 cells).
# Varied: hidden_size in {64, 128, 256, 512} x 3 seeds = 12 cells.
#
# Why three flags (policy + transformer.input + transformer.hidden): the actor
# head in pufferlib/ocean/torch.py:86 is built as Linear(policy.hidden_size, n_actions),
# so if we only change the transformer's hidden_size, the actor's input dim
# (= policy.hidden_size = 256) won't match the transformer's output dim
# (= transformer.hidden_size = H). All three must move together. This is a
# full-width "model scale" ablation (encoder + attention together).
#
# Co-player config is UNTOUCHED (stays at hidden_size=256 to match the
# 2e029h15 checkpoint); only the ego model scales.
# num_heads=4 is unchanged: head_dim = hidden_size/4 in {16, 32, 64, 128}.
#
# Submit: sbatch scripts/adaptive/cluster_hiddensize_ablation.sh

CONFIGS=(
  "64 42" "64 43" "64 44"
  "128 42" "128 43" "128 44"
  "256 42" "256 43" "256 44"
  "512 42" "512 43" "512 44"
)

read -r HIDDEN_SIZE SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Fixed (matches cluster_coplayer_grid_k234.sh recipe for the 2e029h15 / k=4 cells)
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
TOTAL_TIMESTEPS=3000000000
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))
COLLISION_PENALTY_EGO=-0.5
OFFROAD_PENALTY_EGO=-0.5
NUM_WORKERS=32; NUM_ENVS=32
MINIBATCH_MULTIPLIER=50
MAX_MINIBATCH_SIZE=$((50 * HORIZON))

# Shared tag for the whole ablation; per-run hidden_size + seed live in wandb
# config via CLI flags (per the sweep-tag convention).
TAG="hidden_size_ablation_k4_e010"

echo "[hsize_abl] task=$SLURM_ARRAY_TASK_ID  hidden=$HIDDEN_SIZE  seed=$SEED  partner=$PARTNER_ID  e_ub=$ENTROPY_UB  k=$K_SCENARIOS  horizon=$HORIZON  tag=$TAG"

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e

   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate

   export WANDB_MODE=online
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export PUFFER_TRANSFORMER_LEGACY_EVAL=1

   nice -n 19 python scripts/gpu_heartbeat.py &
   HEARTBEAT_PID=\$!

   xvfb-run -a puffer train puffer_adaptive_drive \
     --wandb --wandb-project adaptive_aligned_v2 \
     --tag $TAG \
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
     --train.checkpoint-interval 10 \
     --train.render-interval 30 \
     --train.seed $SEED \
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
     --env.entropy-curriculum-enabled False \
     --eval.map-dir              resources/drive/binaries/nuplan_hard \
     --eval.num-maps             540 \
     --eval.human-replay-eval    True \
     --eval.human-replay-num-rollouts 10 \
     --eval.human-replay-num-maps    540 \
     --eval.human-replay-num-agents  540 \
     --eval.eval-interval 10

   kill \$HEARTBEAT_PID
 "
