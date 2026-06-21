#!/bin/bash
#SBATCH --job-name=eval540
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-14
#
# Re-eval all 15 final checkpoints (k2..k6 x 3 seeds) at 540 maps x 20 rollouts,
# logging the per_map_summary table to each run's own wandb + a local CSV.
# l40s (48GB) fits the legacy forward even for k6 (540 agents, horizon 1206 ~28GB).
#
# Array idx -> "wid k seed iter":
CONFIGS=(
  "ofm0rrbm 2 42 228"
  "r2wr75ay 2 43 170"
  "vx2g4pcg 2 44 140"
  "0qsa6hku 3 42 152"
  "lkgv9a0b 3 43 152"
  "pbvf72ym 3 44 100"
  "qxw6c0jh 4 42 110"
  "ufmegw4l 4 43 114"
  "jsckmpha 4 44 80"
  "jc264zfr 5 42 365"
  "nlzthr49 5 43 365"
  "i09bkafr 5 44 365"
  "f3p7nms8 6 42 304"
  "r5dgbtmg 6 43 304"
  "macfatw8 6 44 304"
)
# Submit: sbatch scripts/adaptive/cluster_eval540_all.sh

read -r WID K SEED ITER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540] task=$SLURM_ARRAY_TASK_ID wid=$WID k=$K seed=$SEED iter=$ITER"

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e
   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate
   export WANDB_MODE=online
   export PUFFER_TRANSFORMER_LEGACY_EVAL=1
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   xvfb-run -a python scripts/adaptive/eval_final_540.py \
     --wid $WID --k $K --seed $SEED --iter $ITER \
     --num-maps 540 --num-agents 540 --num-rollouts 20 \
     --timeout-sec 18000
 "
