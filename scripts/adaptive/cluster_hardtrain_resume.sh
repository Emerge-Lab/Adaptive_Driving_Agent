#!/bin/bash
#SBATCH --job-name=hardtrain_resume
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=256GB
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-5

# Curriculum continuation: resume trained k=4/e=0.10 (h=256) checkpoints for
# +1B steps (3B -> 4B). Arm F trains on interaction-dense maps only
# (nuplan_201_hardtrain, 753 maps); arm U is the matched control on the
# original uniform pool. Cells 0-2 = F seeds 42/43/44; cells 3-5 = U.

CONFIGS=(
  "qxw6c0jh 42 hard"
  "ufmegw4l 43 hard"
  "jsckmpha 44 hard"
  "qxw6c0jh 42 uniform"
  "ufmegw4l 43 uniform"
  "jsckmpha 44 uniform"
)

read -r WID SEED ARM <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

EXP_DIR="experiments/puffer_adaptive_drive_${WID}"
LOAD_CKPT=$(ls -1 ${EXP_DIR}/model_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$LOAD_CKPT" ]; then
  echo "[hardtrain_resume] FATAL: no checkpoint in ${EXP_DIR}" >&2
  exit 1
fi

if [ "$ARM" == "hard" ]; then
  MAP_DIR="resources/drive/binaries/nuplan_201_hardtrain"
  NUM_MAPS=753
  TAG="hardtrain_resume_k4_e010"
else
  MAP_DIR="resources/drive/binaries/nuplan_201"
  NUM_MAPS=4999
  TAG="uniform_resume_k4_e010"
fi

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
TOTAL_TIMESTEPS=4000000000
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))
NUM_WORKERS=32; NUM_ENVS=32
MINIBATCH_MULTIPLIER=50
MAX_MINIBATCH_SIZE=$((50 * HORIZON))

echo "[hardtrain_resume] task=$SLURM_ARRAY_TASK_ID arm=$ARM wid=$WID seed=$SEED ckpt=$LOAD_CKPT map_dir=$MAP_DIR num_maps=$NUM_MAPS tag=$TAG"

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
     --load-model-path $LOAD_CKPT \
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
     --env.map-dir $MAP_DIR \
     --env.num-maps $NUM_MAPS \
     --env.scenario-length $SCENARIO_LENGTH \
     --env.k-scenarios $K_SCENARIOS \
     --env.goal-behavior 3 \
     --env.conditioning.type none \
     --env.reward-lane-align $LANE_REWARD \
     --env.reward-vehicle-collision -0.5 \
     --env.reward-offroad-collision -0.5 \
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
