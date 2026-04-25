#!/bin/bash
#SBATCH --job-name=human_align_ablation
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# Human behavior alignment ablation: 6 experiments
# collision_weight_lb × offroad_weight_lb grid
#
# | Exp | collision_weight_lb | offroad_weight_lb |
# |-----|---------------------|-------------------|
# |  0  |        -3           |       -2.0        |
# |  1  |        -3           |       -1.0        |
# |  2  |        -3           |       -0.5        |
# |  3  |        -2           |       -2.0        |
# |  4  |        -2           |       -1.0        |
# |  5  |        -2           |       -0.5        |

# Define the grid
COLLISION_WEIGHTS=(-3 -3 -3 -2 -2 -2)
OFFROAD_WEIGHTS=(-2.0 -1.0 -0.5 -2.0 -1.0 -0.5)

# Get values for this array task
COLLISION_LB=${COLLISION_WEIGHTS[$SLURM_ARRAY_TASK_ID]}
OFFROAD_LB=${OFFROAD_WEIGHTS[$SLURM_ARRAY_TASK_ID]}

# Fixed parameters
DISCOUNT_UB=0.98
SEED=42
NUPLAN_NUM_MAPS=5000

echo "Running experiment $SLURM_ARRAY_TASK_ID: collision_weight_lb=$COLLISION_LB, offroad_weight_lb=$OFFROAD_LB"

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

    puffer train puffer_drive \
      --wandb --wandb-project human-align-ablation \
      --tag ablation_collision${COLLISION_LB}_offroad${OFFROAD_LB} \
      --env.map-dir resources/drive/binaries/nuplan \
      --env.num-maps $NUPLAN_NUM_MAPS \
      --env.conditioning.type all \
      --env.conditioning.collision-weight-lb $COLLISION_LB \
      --env.conditioning.collision-weight-ub 0 \
      --env.conditioning.offroad-weight-lb $OFFROAD_LB \
      --env.conditioning.offroad-weight-ub 0 \
      --env.conditioning.discount-weight-lb 0.8 \
      --env.conditioning.discount-weight-ub $DISCOUNT_UB \
      --policy-architecture Transformer \
      --train.context-length 91 \
      --train.horizon 91 \
      --train.seed $SEED \
      --eval.map-dir resources/drive/binaries/nuplan

    kill \$HEARTBEAT_PID
  "
