#!/bin/bash
#SBATCH --job-name=eval540_demo
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

CONFIGS=(
  "qgzr83aa 4 42 114 2e029h15 128"
  "ax4hvk7w 4 43 114 2e029h15 128"
  "hib8dwoo 4 44 114 2e029h15 128"
  "gmfve041 4 42 114 2e029h15 128"
  "gxm7jb7t 4 43  90 2e029h15 128"
  "kualhrgw 4 44 114 2e029h15 128"
)

read -r WID K SEED ITER PARTNER HIDDEN <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540_demo] task=$SLURM_ARRAY_TASK_ID wid=$WID k=$K seed=$SEED iter=$ITER partner=$PARTNER hidden=$HIDDEN demo=True"

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
     --demo-trial-0 \
     --num-maps 540 --num-agents 540 --num-rollouts 20 \
     --out-dir outputs/eval540_demo \
     --return-dir outputs/eval540_demo \
     --table-prefix eval540_demo_20r \
     --timeout-sec 18000
 "
