#!/bin/bash
#SBATCH --job-name=adaptive_nuplan_tfm
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

# Train adaptive agents on NuPlan with Transformer architecture
# Uses pre-trained NuPlan Transformer co-players with varied conditioning
#
# PREREQUISITE: Train co-players first with scripts/coplayers/nuplan_transformer.sh
# Then update ZIPPED_RUNS with the trained policy paths from wandb

# Co-player policies trained with scripts/coplayers/nuplan_transformer.sh
# Each entry: "policy_path entropy_weight_ub discount_weight_lb"
ZIPPED_RUNS=(
"experiments/puffer_drive_joqbmi4s.pt 0.01 0.2"
"experiments/puffer_drive_0h81rtfi.pt 0 0.4"
# "experiments/puffer_drive_medzmgum.pt 0.1 0.2"
# "experiments/puffer_drive_f4a8yoi9.pt 0.5 0.2"
"experiments/puffer_drive_b1yx43w2.pt 0.01 0.4"
# "experiments/puffer_drive_lv4x8hlt.pt 0.1 0.4"
# "experiments/puffer_drive_j51yz49e.pt 0.5 0.4"
"experiments/puffer_drive_iry1wanp.pt 0 0.6"
"experiments/puffer_drive_kx8bhu3v.pt 0.01 0.6"
"experiments/puffer_drive_js4mf85k.pt 0.1 0.6"
# "experiments/puffer_drive_52u5onve.pt 0.5 0.6"
"experiments/puffer_drive_nu7lkmx4.pt 0 0.8"
"experiments/puffer_drive_f8dpkpbq.pt 0.01 0.8"
"experiments/puffer_drive_zxsxu6z7.pt 0.1 0.8"
# "experiments/puffer_drive_mfahi5bc.pt 0.5 0.8"
)

read -r COPLAYER_PATH ENTROPY_UB DISCOUNT_LB <<< "${ZIPPED_RUNS[$SLURM_ARRAY_TASK_ID]}"

# Fixed values
CONDITION_TYPE="all"
DISCOUNT_UB=1
ENTROPY_LB=0
NUPLAN_NUM_MAPS=5000

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

   puffer train puffer_adaptive_drive --wandb --tag adaptive_nuplan_transformer_4apr \
     --env.map-dir resources/drive/binaries/nuplan \
     --env.num-maps $NUPLAN_NUM_MAPS \
     --env.conditioning.type none \
     --env.co-player-enabled 1 \
     --env.co-player-policy.policy-path $COPLAYER_PATH \
     --env.co-player-policy.architecture Transformer \
     --env.co-player-policy.conditioning.type $CONDITION_TYPE \
     --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
     --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
     --env.co-player-policy.conditioning.entropy-weight-lb $ENTROPY_LB \
     --env.co-player-policy.conditioning.entropy-weight-ub $ENTROPY_UB \
     --rnn-name Transformer \

   kill \$HEARTBEAT_PID
 "
