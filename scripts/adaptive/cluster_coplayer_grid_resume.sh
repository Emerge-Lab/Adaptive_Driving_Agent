#!/bin/bash
#SBATCH --job-name=cpgrid_resume
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:h100:1
#SBATCH --array=7,12,14,16,22
#
# RESUME variant of cluster_coplayer_grid_k234.sh. Same recipe/config table,
# but for sub-target cells that hit the 24h wall (or hung) we continue from
# the latest checkpoint instead of restarting from scratch:
#   --load-model-path <latest model_*.pt>  -> restores weights + Adam state +
#       global_step + cosine LR position (via sibling trainer_state.pt), so the
#       loop `while global_step < total_timesteps` trains only the REMAINDER.
#   --load-id <wandb_id>                    -> wandb resumes the SAME run
#       (continuous history/name/tags); checkpoints land in the same exp dir.
#
# Cells to resume (array index == original k234 task index):
#   7  miku2puk 0.05 k4 s43  wid=bhx6zxn0  iter 80/114
#   12 m2ygolog 0.20 k3 s42  wid=052n7brp  iter 90/152
#   14 m2ygolog 0.20 k3 s44  wid=cei22yc1  iter 130/152
#   16 m2ygolog 0.20 k4 s43  wid=citbzhdc  iter 10/114  (hung on gh015)
#   22 6rauydj2 0.50 k3 s43  wid=blyerjec  iter 140/152
#
# Submit (exclude the bad node that hung task 16):
#   sbatch --exclude=gh015 scripts/adaptive/cluster_coplayer_grid_resume.sh
CONFIGS=(
  "miku2puk 0.05 2 42" "miku2puk 0.05 2 43" "miku2puk 0.05 2 44"
  "miku2puk 0.05 3 42" "miku2puk 0.05 3 43" "miku2puk 0.05 3 44"
  "miku2puk 0.05 4 42" "miku2puk 0.05 4 43" "miku2puk 0.05 4 44"
  "m2ygolog 0.20 2 42" "m2ygolog 0.20 2 43" "m2ygolog 0.20 2 44"
  "m2ygolog 0.20 3 42" "m2ygolog 0.20 3 43" "m2ygolog 0.20 3 44"
  "m2ygolog 0.20 4 42" "m2ygolog 0.20 4 43" "m2ygolog 0.20 4 44"
  "6rauydj2 0.50 2 42" "6rauydj2 0.50 2 43" "6rauydj2 0.50 2 44"
  "6rauydj2 0.50 3 42" "6rauydj2 0.50 3 43" "6rauydj2 0.50 3 44"
  "6rauydj2 0.50 4 42" "6rauydj2 0.50 4 43" "6rauydj2 0.50 4 44"
)

# Resume target: original wandb run id per array index.
declare -A RESUME_WID=( [7]=bhx6zxn0 [12]=052n7brp [14]=cei22yc1 [16]=citbzhdc [22]=blyerjec )

read -r PARTNER_ID ENTROPY_UB K_SCENARIOS SEED <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

WID=${RESUME_WID[$SLURM_ARRAY_TASK_ID]}
EXP_DIR="experiments/puffer_adaptive_drive_${WID}"
# Latest checkpoint (numeric/version sort); its sibling trainer_state.pt carries
# optimizer + global_step + epoch for a true mid-run resume.
LOAD_CKPT=$(ls -1 ${EXP_DIR}/model_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$LOAD_CKPT" ]; then
  echo "[cpgrid_resume] FATAL: no checkpoint found in ${EXP_DIR}" >&2
  exit 1
fi

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

# Per-k tag so co-players group with the existing 0.10 runs at the same k;
# partner + entropy_ub live in wandb config (CLI flags) for sub-grouping.
TAG="ada_k${K_SCENARIOS}_gb3_legacy_eval_fix"

echo "[cpgrid_resume] task=$SLURM_ARRAY_TASK_ID  partner=$PARTNER_ID  e_ub=$ENTROPY_UB  k=$K_SCENARIOS  seed=$SEED  horizon=$HORIZON  tag=$TAG"
echo "[cpgrid_resume] RESUME wid=$WID  ckpt=$LOAD_CKPT"

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
     --load-model-path $LOAD_CKPT \
     --load-id $WID \
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
