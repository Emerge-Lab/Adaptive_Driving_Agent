#!/bin/bash
#SBATCH --job-name=adaptive_womd_tfm
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

# Train adaptive agents on WOMD with Transformer architecture
# Uses pre-trained WOMD Transformer co-players with varied conditioning
#
# PREREQUISITE: Train co-players first with scripts/coplayers/womd_transformer.sh
# Then update ZIPPED_RUNS with the trained policy paths from wandb

# Co-player policies trained with scripts/coplayers/womd_transformer.sh
# Each entry: "policy_path entropy_weight_ub discount_weight_lb"
ZIPPED_RUNS=(
  "experiments/puffer_drive_zagelrzs.pt 0.5 0.8"
  "experiments/puffer_drive_d8kb6hwf.pt 0.1 0.8"
  "experiments/puffer_drive_xdmwezaw.pt 0.01 0.8"
  "experiments/puffer_drive_0cxi9nf8.pt 0 0.8"

  "experiments/puffer_drive_t69evoxz.pt 0.5 0.6"
  "experiments/puffer_drive_yuuod9cn.pt 0.1 0.6"
  "experiments/puffer_drive_436bzeu2.pt 0.01 0.6"
  "experiments/puffer_drive_ct49w01c.pt 0 0.6"

  "experiments/puffer_drive_1e54zwgz.pt 0.5 0.4"
  "experiments/puffer_drive_epupe6sw.pt 0.1 0.4"
  "experiments/puffer_drive_npqu25y1.pt 0.01 0.4"
  "experiments/puffer_drive_v9urng8s.pt 0 0.4"

  "experiments/puffer_drive_fugsjie2.pt 0.5 0.2"
  "experiments/puffer_drive_iejlfoo7.pt 0.1 0.2"
  "experiments/puffer_drive_rl7e091t.pt 0.01 0.2"
  "experiments/puffer_drive_vztz9mmh.pt 0 0.2"
)

read -r COPLAYER_PATH ENTROPY_UB DISCOUNT_LB <<< "${ZIPPED_RUNS[$SLURM_ARRAY_TASK_ID]}"

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

   nice -n 19 python scripts/gpu_heartbeat.py &
   HEARTBEAT_PID=\$!

   puffer train puffer_adaptive_drive --wandb --tag adaptive_womd_transformer_new \
     --env.num-maps 10000 \
     --env.conditioning.type none \
     --env.co-player-enabled 1 \
     --env.co-player-policy.policy-path $COPLAYER_PATH \
     --env.co-player-policy.conditioning.type $CONDITION_TYPE \
     --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
     --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
     --env.co-player-policy.conditioning.entropy-weight-lb $ENTROPY_LB \
     --env.co-player-policy.conditioning.entropy-weight-ub $ENTROPY_UB \
     --policy-architecture Transformer \
     --rnn-name Transformer

   kill \$HEARTBEAT_PID
 "
