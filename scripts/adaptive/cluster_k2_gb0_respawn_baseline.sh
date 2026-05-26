#!/bin/bash
#SBATCH --job-name=k2_gb0_baseline
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-2

# k=2 gb=0 (respawn) baseline. 3 seeds x 1 (LR, ent_coef) cell = 3 array tasks.
# Companion to cluster_k2_gb3_lr_ent_sweep.sh — same hyperparams except
# goal_behavior=0 (respawn) instead of 3 (trial). Used to isolate whether the
# trial structure itself is the bottleneck for env/score, separate from
# optimization issues.
#
# Submit: sbatch scripts/adaptive/cluster_k2_gb0_respawn_baseline.sh

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Fixed partner: p010 (entropy_ub=0.10) — matches the k2 trial sweep
PARTNER_ID=2e029h15
ENTROPY_UB=0.10
COPLAYER_PATH="experiments/puffer_drive_${PARTNER_ID}.pt"

# Optimizer midpoint of the gb=3 LR x ent sweep
LR=3e-3
ENT_COEF=0.005

# Fixed
GAMMA=0.995
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.05
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
TOTAL_TIMESTEPS=3000000000   # 3B from scratch, matches k2 sweep

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))   # 402
COLLISION_PENALTY_EGO=-0.5
OFFROAD_PENALTY_EGO=-0.5

NUM_WORKERS=32; NUM_ENVS=32
MINIBATCH_MULTIPLIER=50                       # matches k2 sweep
MAX_MINIBATCH_SIZE=20100

TAG="baseline_k2_gb0_respawn_lane0.05"

echo "[baseline gb0] task=$SLURM_ARRAY_TASK_ID  seed=$SEED  tag=$TAG"

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
     --wandb --wandb-project adaptive_aligned_v2 \
     --tag $TAG \
     --policy-architecture Transformer --rnn-name Transformer \
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
     --env.goal-behavior 0 \
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
