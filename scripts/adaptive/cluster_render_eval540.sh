#!/bin/bash
#SBATCH --job-name=rendereval540
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
# Same as cluster_render_grid.sh but for the 0.10 co-player column (the original
# 15-run sweep, ego trained vs partner 2e029h15). These complete the 4-partner
# grid; their per-map CSVs live in outputs/eval540/ (not eval540_grid/).
# Renders 20 videos/cell: 10 most-regressed + 10 best-adapting by ada_delta.
# mp4s -> outputs/videos_grid/{wid}/ (wids are unique across both columns).
#
# CONFIGS and MAPIDS are PARALLEL arrays. iters match what eval540_all.sh used.
# Submit: sbatch scripts/adaptive/cluster_render_eval540.sh
CONFIGS=(
  "ofm0rrbm 2 42 228 2e029h15" "r2wr75ay 2 43 170 2e029h15" "vx2g4pcg 2 44 140 2e029h15"
  "0qsa6hku 3 42 152 2e029h15" "lkgv9a0b 3 43 152 2e029h15" "pbvf72ym 3 44 100 2e029h15"
  "qxw6c0jh 4 42 110 2e029h15" "ufmegw4l 4 43 114 2e029h15" "jsckmpha 4 44 80 2e029h15"
  "jc264zfr 5 42 365 2e029h15" "nlzthr49 5 43 365 2e029h15" "i09bkafr 5 44 365 2e029h15"
  "f3p7nms8 6 42 304 2e029h15" "r5dgbtmg 6 43 304 2e029h15" "macfatw8 6 44 304 2e029h15"
)
MAPIDS=(
  "495 452 280 453 225 244 388 204 53 121 299 65 49 30 460 535 5 445 123 451"   # ofm0rrbm k2 s42
  "121 32 455 178 7 72 100 146 398 33 60 82 163 46 535 329 20 232 136 188"   # r2wr75ay k2 s43
  "49 82 163 286 361 204 87 305 488 529 271 167 181 526 9 329 47 121 457 298"   # vx2g4pcg k2 s44
  "468 46 30 78 87 531 401 460 64 320 238 355 333 496 472 290 493 376 475 169"   # 0qsa6hku k3 s42
  "104 233 452 177 250 80 96 425 465 189 309 443 125 34 128 268 468 391 220 240"   # lkgv9a0b k3 s43
  "526 273 46 443 362 97 108 252 268 317 533 47 179 519 96 488 379 152 76 266"   # pbvf72ym k3 s44
  "85 65 271 454 328 457 92 303 329 355 204 283 496 83 486 425 84 244 41 152"   # qxw6c0jh k4 s42
  "92 41 252 376 88 533 416 404 271 278 65 230 91 232 386 66 329 453 454 4"   # ufmegw4l k4 s43
  "84 229 108 298 344 218 515 334 58 230 271 121 224 30 284 457 318 355 195 66"   # jsckmpha k4 s44
  "169 187 58 412 283 498 366 451 79 457 317 530 331 333 475 179 320 391 395 450"   # jc264zfr k5 s42
  "393 53 34 437 41 473 3 30 88 401 24 345 523 100 446 178 187 91 238 358"   # nlzthr49 k5 s43
  "488 331 34 450 189 236 435 303 171 363 464 106 158 296 42 219 221 297 391 402"   # i09bkafr k5 s44
  "91 169 34 345 376 97 53 329 496 435 531 255 430 469 472 72 248 450 145 425"   # f3p7nms8 k6 s42
  "100 487 244 44 169 394 401 177 191 424 400 314 1 106 158 165 297 464 475 535"   # r5dgbtmg k6 s43
  "246 84 445 164 393 108 252 383 88 77 384 388 415 282 12 144 80 30 379 334"   # macfatw8 k6 s44
)

read -r WID K SEED ITER PARTNER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MAP_IDS="${MAPIDS[$SLURM_ARRAY_TASK_ID]}"
N_STEPS=$((K * 201))
ITER6=$(printf "%06d" "$ITER")
OUT="outputs/videos_grid/${WID}"
echo "[rendereval540] task=$SLURM_ARRAY_TASK_ID partner=$PARTNER wid=$WID k=$K seed=$SEED iter=$ITER n_steps=$N_STEPS"
echo "[rendereval540] map_ids: $MAP_IDS"

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
