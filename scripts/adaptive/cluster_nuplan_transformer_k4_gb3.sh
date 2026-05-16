#!/bin/bash
#SBATCH --job-name=ada_k4_gb3
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

# k=4 gb=3 (GOAL_TRIAL) adaptive sweep: 4 partners × 3 seeds = 12 tasks.
# Each array task is ONE partner × ONE seed on ONE GPU with nw=32.
#
# Array indexing: TASK_ID = partner_idx * 3 + seed_idx
#   partner_idx ∈ {0..3}  → PARTNERS[partner_idx]
#   seed_idx    ∈ {0..2}  → SEEDS[seed_idx]
#
# Submit: sbatch scripts/adaptive/cluster_nuplan_transformer_k4_gb3.sh

# (label, partner_id, entropy_ub)
PARTNERS=(
  "p005   miku2puk   0.05"
  "p010   2e029h15   0.10"
  "p020   m2ygolog   0.20"
  "p050   6rauydj2   0.50"
)
SEEDS=(42 43 44)

PARTNER_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
read -r LABEL PARTNER_ID ENTROPY_UB <<< "${PARTNERS[$PARTNER_IDX]}"
SEED=${SEEDS[$SEED_IDX]}
COPLAYER_PATH="experiments/puffer_drive_${PARTNER_ID}.pt"

# Fixed
GAMMA=0.995
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.025
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
TOTAL_TIMESTEPS=2000000000   # 2B

K_SCENARIOS=4
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))   # 804

NUM_WORKERS=32; NUM_ENVS=32
MINIBATCH_MULTIPLIER=25                       # minibatch_size = 25 * 804 = 20100
MAX_MINIBATCH_SIZE=20100

TAG="ada_k4_gb3_${LABEL}_${PARTNER_ID}_s${SEED}_g${GAMMA}_lane${LANE_REWARD}_nw${NUM_WORKERS}"

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

   nice -n 19 python scripts/gpu_heartbeat.py &
   HEARTBEAT_PID=\$!

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
     --eval.map-dir              resources/drive/binaries/nuplan_hard \
     --eval.num-maps             64 \
     --eval.human-replay-eval    True \
     --eval.human-replay-num-rollouts 40 \
     --eval.human-replay-num-maps    64 \
     --eval.human-replay-num-agents  64 \
     --eval.eval-interval 10

   kill \$HEARTBEAT_PID
 "
