#!/bin/bash
#SBATCH --job-name=coplayer_womd_tfm_ablation
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

# Ablation study on Transformer architecture parameters
# Varies: context length (1, 32, 91), learning rate (0.003, 0.0003), input_size (64, 128)
# Uses default entropy/discount values with "all" conditioning

# Grid: 3 context × 2 learning rate × 2 input_size = 12 configurations
# Format: "CTX LR INPUT_SIZE"
ZIPPED_RUNS=(
  # ctx=1
  "1 0.003 64"
  "1 0.003 128"
  "1 0.0003 64"
  "1 0.0003 128"

  # ctx=32
  "32 0.003 64"
  "32 0.003 128"
  "32 0.0003 64"
  "32 0.0003 128"

  # ctx=91
  "91 0.003 64"
  "91 0.003 128"
  "91 0.0003 64"
  "91 0.0003 128"
)

read -r CTX LR INPUT_SIZE <<< "${ZIPPED_RUNS[$SLURM_ARRAY_TASK_ID]}"

# Calculate minibatch_size: must be <= batch_size (8192 * CTX) and divisible by CTX
# Using multiplier 352 (same as 32032/91) for consistency
MINIBATCH_SIZE=$((CTX * 352))

# Fixed conditioning values (matching drive.ini defaults)
CONDITION_TYPE="all"
ENTROPY_UB=0.001
ENTROPY_LB=0
DISCOUNT_UB=0.98
DISCOUNT_LB=0.80

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e

   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate

   nice -n 19 python scripts/gpu_heartbeat.py &
   HEARTBEAT_PID=\$!

   puffer train puffer_drive --wandb --wandb-project ada_new_coplayers --tag coplayer_womd_transformer_ablation \
     --env.num-maps 10000 \
     --env.conditioning.type $CONDITION_TYPE \
     --env.conditioning.entropy-weight-lb $ENTROPY_LB \
     --env.conditioning.entropy-weight-ub $ENTROPY_UB \
     --env.conditioning.discount-weight-lb $DISCOUNT_LB \
     --env.conditioning.discount-weight-ub $DISCOUNT_UB \
     --policy-architecture Transformer \
     --policy.input-size $INPUT_SIZE \
     --train.context-length $CTX \
     --train.horizon $CTX \
     --train.learning-rate $LR \
     --train.minibatch-size $MINIBATCH_SIZE \
     --train.checkpoint-interval 50

   kill \$HEARTBEAT_PID
 "
