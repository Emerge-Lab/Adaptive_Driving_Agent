#!/bin/bash
#SBATCH --job-name=eval540_curr
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-11

# Eval for the curriculum continuation arms (cells 0-8: hard/uniform/frontier
# x 3 seeds, iter 152 = 4B) and the e_ub=0.001 entropy anchors (cells 9-11,
# iter 114 = 3B). Standard 540x20 human-replay suite.
# Format: "wid k seed iter outdir"

CONFIGS=(
  "7m6lu6z1 4 42 152 eval540_curriculum"
  "s4h7at8q 4 43 152 eval540_curriculum"
  "l9ljqxar 4 44 152 eval540_curriculum"
  "yd33nuj2 4 42 152 eval540_curriculum"
  "vow739q9 4 43 152 eval540_curriculum"
  "zcijpff5 4 44 152 eval540_curriculum"
  "h29ja02n 4 42 152 eval540_curriculum"
  "fc2lp7x8 4 43 152 eval540_curriculum"
  "afimi1r6 4 44 152 eval540_curriculum"
  "wyoe194j 4 42 114 eval540_e0001"
  "x3vxow3u 4 43 114 eval540_e0001"
  "uzfye8l1 4 44 114 eval540_e0001"
)

read -r WID K SEED ITER OUTDIR <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540_curr] task=$SLURM_ARRAY_TASK_ID wid=$WID k=$K seed=$SEED iter=$ITER out=$OUTDIR"

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
     --out-dir outputs/$OUTDIR \
     --return-dir outputs/$OUTDIR \
     --table-prefix ${OUTDIR}_20r \
     --timeout-sec 18000
 "
