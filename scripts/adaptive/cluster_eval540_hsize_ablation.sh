#!/bin/bash
#SBATCH --job-name=eval540_hsize
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-8
#
# Human-replay eval on nuplan_hard (540 maps, 20 rollouts) for the
# hidden_size ablation (array 12004810). k=4, partner=2e029h15 (e_ub=0.10)
# fixed; only hidden_size and seed vary.
#
# Cells (9 = 3 hidden × 3 seeds; h=512 skipped: OOM'd during training).
# cell6 (h=256, s=42) uses iter=80 (partial run, hit OOM later).
# Format: "wid k seed iter partner hidden"

CONFIGS=(
  "gcknljp3 4 42 114 2e029h15  64"
  "zrps018m 4 43 114 2e029h15  64"
  "oxxiy4yd 4 44 114 2e029h15  64"
  "qgzr83aa 4 42 114 2e029h15 128"
  "ax4hvk7w 4 43 114 2e029h15 128"
  "hib8dwoo 4 44 114 2e029h15 128"
  "k29sr0rw 4 42  80 2e029h15 256"   # partial (training OOM'd)
  "pv53oner 4 43 114 2e029h15 256"
  "2rkkkxr3 4 44 114 2e029h15 256"
)
# Submit: sbatch scripts/adaptive/cluster_eval540_hsize_ablation.sh

read -r WID K SEED ITER PARTNER HIDDEN <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540_hsize] task=$SLURM_ARRAY_TASK_ID partner=$PARTNER wid=$WID k=$K seed=$SEED iter=$ITER hidden=$HIDDEN"

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
     --hidden-size $HIDDEN \
     --num-maps 540 --num-agents 540 --num-rollouts 20 \
     --out-dir outputs/eval540_hsize \
     --return-dir outputs/eval540_hsize \
     --table-prefix eval540_hsize_20r \
     --timeout-sec 18000
 "
