#!/bin/bash
#SBATCH --job-name=rendergrid2
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
# Batch-2 render: the 6 re-run/resubmit grid cells that completed after the
# original 39-cell render (cluster_render_grid.sh). Same 20-video recipe
# (10 most-regressed + 10 best-adapting maps by per-map ada_delta). Map-id lists
# were computed from outputs/eval540_grid/per_map_*_{wid}.csv (eval batch-2).
# All 6 are k3/k4 so the legacy forward fits in one chunk. mp4s -> videos_grid/{wid}/.
#
# CONFIGS and MAPIDS are PARALLEL arrays.
# Submit: sbatch scripts/adaptive/cluster_render_grid_batch2.sh
CONFIGS=(
  "bhx6zxn0 4 43 114 miku2puk"
  "9qewt905 4 44 114 miku2puk"
  "052n7brp 3 42 152 m2ygolog"
  "cei22yc1 3 44 152 m2ygolog"
  "citbzhdc 4 43 114 m2ygolog"
  "blyerjec 3 43 152 6rauydj2"
)
MAPIDS=(
  "215 218 5 70 344 49 64 418 220 146 66 355 65 383 533 157 445 535 128 72"   # bhx6zxn0 k4 s43
  "471 204 199 468 507 88 121 194 92 97 435 357 283 298 220 504 107 303 535 391"   # 9qewt905 k4 s44
  "72 121 424 529 278 300 367 401 271 246 66 404 204 83 451 388 383 235 309 435"   # 052n7brp k3 s42
  "379 67 124 249 235 157 181 17 273 450 526 283 276 496 97 72 333 479 228 318"   # cei22yc1 k3 s44
  "366 507 80 146 479 328 220 252 334 495 58 224 383 345 121 107 451 435 47 535"   # citbzhdc k4 s43
  "229 317 94 205 237 280 469 445 77 107 320 337 391 345 41 179 261 446 53 1"   # blyerjec k3 s43
)

read -r WID K SEED ITER PARTNER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MAP_IDS="${MAPIDS[$SLURM_ARRAY_TASK_ID]}"
N_STEPS=$((K * 201))
ITER6=$(printf "%06d" "$ITER")
OUT="outputs/videos_grid/${WID}"
echo "[rendergrid2] task=$SLURM_ARRAY_TASK_ID partner=$PARTNER wid=$WID k=$K seed=$SEED iter=$ITER n_steps=$N_STEPS"
echo "[rendergrid2] map_ids: $MAP_IDS"

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e
   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate
   export PUFFER_TRANSFORMER_LEGACY_EVAL=1
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

   xvfb-run -a python tests/render_specific_maps.py \
     --checkpoint experiments/puffer_adaptive_drive_${WID}/model_puffer_adaptive_drive_${ITER6}.pt \
     --info       experiments/puffer_adaptive_drive_${WID}/info.json \
     --map-ids    $MAP_IDS \
     --n-steps    $N_STEPS \
     --map-dir    resources/drive/binaries/nuplan_hard \
     --out        $OUT
 "
