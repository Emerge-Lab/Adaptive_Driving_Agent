#!/bin/bash
#SBATCH --job-name=eval540grid2
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-5
#
# Batch-2 of the eval540 grid: the 6 cells that were re-run/resubmitted after the
# original 39-cell batch (cluster_eval540_grid.sh). All reached their per-k iter
# target (verified by checkpoint). Same recipe as batch-1 -> outputs/eval540_grid/.
# Config = "wid k seed iter partner" (partner is for the log line only).
CONFIGS=(
  "bhx6zxn0 4 43 114 miku2puk"   # re-run 11287601_7
  "9qewt905 4 44 114 miku2puk"   # resubmit 11131115_8
  "052n7brp 3 42 152 m2ygolog"   # re-run 11287601_12
  "cei22yc1 3 44 152 m2ygolog"   # re-run 11287601_14
  "citbzhdc 4 43 114 m2ygolog"   # re-run 11287601_16
  "blyerjec 3 43 152 6rauydj2"   # re-run 11287601_22
)
# Submit: sbatch scripts/adaptive/cluster_eval540_grid_batch2.sh

read -r WID K SEED ITER PARTNER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540grid2] task=$SLURM_ARRAY_TASK_ID partner=$PARTNER wid=$WID k=$K seed=$SEED iter=$ITER"

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
     --out-dir outputs/eval540_grid \
     --timeout-sec 18000
 "
