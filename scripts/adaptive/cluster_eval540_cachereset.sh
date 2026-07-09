#!/bin/bash
#SBATCH --job-name=eval540_creset
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

# Memory-ablation causal control: same eval as eval540_return for the headline
# cells, but with the ego K/V cache reset at every trial boundary
# (RECOVERY_CACHE_RESET_PER_SCENARIO=1). If the per-trial return curve
# flattens, cross-trial memory is causally established as the driver of
# adaptation. Cells 0-2 = e_ub 0.10 / k4; cells 3-5 = e_ub 0.20 / k4.
# Format: "wid k seed iter"

CONFIGS=(
  "qxw6c0jh 4 42 110"
  "ufmegw4l 4 43 114"
  "jsckmpha 4 44 80"
  "ftxa55g3 4 42 114"
  "citbzhdc 4 43 114"
  "c0k9uqhc 4 44 114"
)

read -r WID K SEED ITER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540_creset] task=$SLURM_ARRAY_TASK_ID wid=$WID k=$K seed=$SEED iter=$ITER cache_reset=1"

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e
   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate
   export WANDB_MODE=disabled
   export PUFFER_TRANSFORMER_LEGACY_EVAL=1
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export RECOVERY_CACHE_RESET_PER_SCENARIO=1
   xvfb-run -a python scripts/adaptive/eval_final_540.py \
     --wid $WID --k $K --seed $SEED --iter $ITER \
     --num-maps 540 --num-agents 540 --num-rollouts 20 \
     --out-dir outputs/eval540_cachereset \
     --return-dir outputs/eval540_cachereset \
     --table-prefix eval540_creset_20r \
     --no-wandb \
     --timeout-sec 18000
 "
