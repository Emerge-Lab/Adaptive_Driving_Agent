#!/bin/bash
#SBATCH --job-name=k56_memsafe
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=600GB
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-5
#
# k5/k6 MEMORY-SAFE retrain. The earlier k5/k6 runs OOM'd on GPU during the
# legacy full-context forward_eval (horizon 1005/1206 over the full rollout
# agent batch → 23GB+ alloc) and never saved checkpoints. Reduce the rollout
# agent batch (num_envs 32->8) and the eval agent batch (540->270) so the
# legacy forward fits in 80GB. 3 seeds (42,43,44) each → 6 array tasks.
#
# Array → (k_scenarios, seed):
CONFIGS=( "5 42" "5 43" "5 44" "6 42" "6 43" "6 44" )
# Submit: sbatch scripts/adaptive/cluster_k56_gb3_memsafe.sh

read -r K_SCENARIOS SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

LR=3e-3
ENT_COEF=0.005
PARTNER_ID=2e029h15
ENTROPY_UB=0.10
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

# Memory-safe knobs (vs the standard k-sweep's nw=32 nv=32): 8 rollout envs to
# shrink the legacy-forward batch. workers=8 so num_envs is divisible by workers
# (num_envs % num_workers must be 0, else vector.make raises APIUsageError).
NUM_WORKERS=8; NUM_ENVS=8
MINIBATCH_MULTIPLIER=25
MAX_MINIBATCH_SIZE=$((25 * HORIZON))
EVAL_NUM_AGENTS=270

TAG="ada_k${K_SCENARIOS}_gb3_legacy_eval_fix"

echo "[k56_memsafe] task=$SLURM_ARRAY_TASK_ID  k=$K_SCENARIOS  seed=$SEED  horizon=$HORIZON  nv=$NUM_ENVS  eval_agents=$EVAL_NUM_AGENTS  tag=$TAG"

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
     --vec.num-workers $NUM_WORKERS --vec.num-envs $NUM_ENVS --vec.batch-size 8 \
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
     --eval.num-maps             $EVAL_NUM_AGENTS \
     --eval.human-replay-eval    True \
     --eval.human-replay-num-rollouts 10 \
     --eval.human-replay-num-maps    $EVAL_NUM_AGENTS \
     --eval.human-replay-num-agents  $EVAL_NUM_AGENTS \
     --eval.eval-interval 10

   kill \$HEARTBEAT_PID
 "
