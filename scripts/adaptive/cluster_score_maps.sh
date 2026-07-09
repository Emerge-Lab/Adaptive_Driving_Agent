#!/bin/bash
#SBATCH --job-name=score_maps
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-29

WIDS=("qxw6c0jh 42 110" "ufmegw4l 43 114" "jsckmpha 44 80")

CHUNK=$((SLURM_ARRAY_TASK_ID % 10))
WIDX=$((SLURM_ARRAY_TASK_ID / 10))
read -r WID SEED ITER <<< "${WIDS[$WIDX]}"

CHUNK_DIR=$(printf "resources/drive/binaries/nuplan_201_chunk%02d" "$CHUNK")
N_MAPS=$(ls "$CHUNK_DIR" | wc -l)
OUT_DIR=$(printf "outputs/map_scoring/chunk%02d_%s" "$CHUNK" "$WID")

echo "[score_maps] task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK wid=$WID seed=$SEED iter=$ITER n_maps=$N_MAPS"

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
   xvfb-run -a python scripts/adaptive/eval_final_540.py \
     --wid $WID --k 4 --seed $SEED --iter $ITER \
     --map-dir $CHUNK_DIR \
     --num-maps $N_MAPS --num-agents $N_MAPS --num-rollouts 5 \
     --out-dir $OUT_DIR \
     --return-dir $OUT_DIR \
     --table-prefix score_maps \
     --no-wandb \
     --timeout-sec 9000
 "
