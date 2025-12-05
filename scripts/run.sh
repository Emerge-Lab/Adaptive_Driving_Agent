#!/bin/bash
#SBATCH --job-name=puffer_drive
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --array=0-15

# Define configurations for each array task ID
# discount_weight_lb: 0.2, 0.4, 0.6, 0.8
# entropy_weight_ub: 0, 0.01, 0.1, 0.5
DISCOUNT_LBS=(0.2 0.2 0.2 0.2 0.4 0.4 0.4 0.4 0.6 0.6 0.6 0.6 0.8 0.8 0.8 0.8)
ENTROPY_UBS=(0 0.01 0.1 0.5 0 0.01 0.1 0.5 0 0.01 0.1 0.5 0 0.01 0.1 0.5)

DISCOUNT_LB=${DISCOUNT_LBS[$SLURM_ARRAY_TASK_ID]}
ENTROPY_UB=${ENTROPY_UBS[$SLURM_ARRAY_TASK_ID]}

# Fixed values
CONDITION_TYPE="all"
DISCOUNT_UB=1
ENTROPY_LB=0

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e

   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate

   puffer train puffer_drive --wandb --env.num-maps 1000 \
     --env.conditioning.type $CONDITION_TYPE \
     --env.conditioning.discount-weight-lb $DISCOUNT_LB \
     --env.conditioning.discount-weight-ub $DISCOUNT_UB \
     --env.conditioning.entropy-weight-lb $ENTROPY_LB \
     --env.conditioning.entropy-weight-ub $ENTROPY_UB
 "
