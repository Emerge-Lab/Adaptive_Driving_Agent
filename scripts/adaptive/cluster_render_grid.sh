#!/bin/bash
#SBATCH --job-name=rendergrid
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-38
#
# Render 20 adaptation videos per grid cell: the 10 best-adapting + 10 most-
# regressed maps by per-map ada_delta (from outputs/eval540_grid CSVs).
# Eval-faithful (use_all_maps + human-replay, legacy full-context forward),
# k-trial episodes so trial-0 -> trial-last is visible in one clip. Each cell
# writes its mp4s into outputs/videos_grid/{wid}/ (no shared-CWD filename race).
# Runs on l40s/priority so it does not compete with the h100 training resumes.
#
# CONFIGS and MAPIDS are PARALLEL arrays (same 39 cells as cluster_eval540_grid.sh).
# Map-id lists are precomputed (10 most-regressed then 10 best, per cell).
# The 6 in-flight cells (5 re-runs + resubmit-8) get a follow-up array later.
#
# Submit: sbatch scripts/adaptive/cluster_render_grid.sh
CONFIGS=(
  # miku2puk  e_ub=0.05
  "6opvas42 2 42 228 miku2puk" "3dlsmo4v 2 43 228 miku2puk" "6gzqx4gj 2 44 228 miku2puk"
  "io7kp2cq 3 42 152 miku2puk" "xr137mjs 3 43 152 miku2puk" "rf4p3hy6 3 44 152 miku2puk"
  "9gc19bcy 4 42 114 miku2puk"
  "246wih6m 5 42 365 miku2puk" "w1u39swd 5 43 365 miku2puk" "f754lim2 5 44 365 miku2puk"
  "ijwa3y93 6 42 304 miku2puk" "hk5gtm99 6 43 304 miku2puk" "yl6tfvb0 6 44 304 miku2puk"
  # m2ygolog  e_ub=0.20
  "obrwxqqy 2 42 228 m2ygolog" "gesyc4j9 2 43 228 m2ygolog" "s790xh0d 2 44 228 m2ygolog"
  "n46bkreg 3 43 152 m2ygolog"
  "ftxa55g3 4 42 114 m2ygolog" "c0k9uqhc 4 44 114 m2ygolog"
  "72dilduo 5 42 365 m2ygolog" "06uc7cis 5 43 365 m2ygolog" "e35ds248 5 44 365 m2ygolog"
  "0r5j0p8y 6 42 304 m2ygolog" "osjfnxz0 6 43 304 m2ygolog" "w8sbas7o 6 44 304 m2ygolog"
  # 6rauydj2  e_ub=0.50
  "cgq46nnk 2 42 228 6rauydj2" "c8yihf5o 2 43 220 6rauydj2" "yw4mao1d 2 44 228 6rauydj2"
  "x4cskyot 3 42 152 6rauydj2" "1eqwuq6m 3 44 152 6rauydj2"
  "m4ibxlhu 4 42 114 6rauydj2" "yu0vk259 4 43 114 6rauydj2" "7s0p1b8q 4 44 114 6rauydj2"
  "hzttkj82 5 42 365 6rauydj2" "fkufm4ol 5 43 365 6rauydj2" "071oqqu5 5 44 365 6rauydj2"
  "p4ltvhdf 6 42 304 6rauydj2" "9ti3ocxm 6 43 304 6rauydj2" "ly30vz8m 6 44 304 6rauydj2"
)
MAPIDS=(
  "455 230 329 260 386 121 345 314 337 171 268 240 282 91 194 395 10 495 237 286"   # 6opvas42 k2 s42
  "320 230 74 4 280 404 124 121 437 507 47 35 5 66 182 393 425 107 300 465"   # 3dlsmo4v k2 s43
  "47 228 246 368 95 195 31 157 479 215 425 41 456 298 427 492 454 22 42 45"   # 6gzqx4gj k2 s44
  "35 232 273 495 5 457 492 313 91 209 386 148 366 78 271 398 368 400 413 475"   # io7kp2cq k3 s42
  "354 5 113 331 461 240 457 104 299 263 205 309 427 533 80 90 379 413 530 220"   # xr137mjs k3 s43
  "76 92 451 398 244 290 533 58 317 298 1 107 42 238 416 322 457 465 425 493"   # rf4p3hy6 k3 s44
  "280 65 121 130 355 471 47 21 58 269 418 5 34 379 487 194 204 445 30 152"   # 9gc19bcy k4 s42
  "235 72 298 376 435 94 386 169 355 383 224 219 42 297 402 158 296 106 464 475"   # 246wih6m k5 s42
  "88 189 233 340 454 100 106 324 290 433 427 473 501 391 313 52 337 358 442 465"   # w1u39swd k5 s43
  "298 100 319 1 278 430 446 507 309 498 303 32 453 67 271 283 355 228 449 97"   # f754lim2 k5 s44
  "281 3 357 345 187 34 496 91 437 90 65 148 236 383 474 504 322 425 224 298"   # ijwa3y93 k6 s42
  "58 88 460 7 30 83 533 135 355 425 387 42 220 230 475 313 320 121 250 445"   # hk5gtm99 k6 s43
  "1 357 457 34 506 58 152 169 345 100 46 285 208 377 254 418 472 454 72 391"   # yl6tfvb0 k6 s44
  "451 121 317 329 388 465 5 246 85 125 492 355 144 300 303 313 490 337 296 391"   # obrwxqqy k2 s42
  "30 88 171 271 531 170 474 191 1 181 188 333 307 314 404 515 236 329 252 395"   # gesyc4j9 k2 s43
  "331 450 488 136 235 74 191 531 204 215 314 85 287 178 357 157 195 317 379 47"   # s790xh0d k2 s44
  "121 204 22 220 47 194 188 30 83 465 95 298 97 368 449 5 232 58 493 446"   # n46bkreg k3 s43
  "66 507 278 266 471 220 194 384 457 96 376 383 22 195 5 303 355 329 188 298"   # ftxa55g3 k4 s42
  "78 229 215 430 331 67 74 379 298 88 354 398 451 355 425 479 278 34 487 204"   # c0k9uqhc k4 s44
  "58 67 345 376 320 187 329 451 235 393 287 530 383 437 452 9 72 125 465 453"   # 72dilduo k5 s42
  "47 121 123 281 282 97 191 285 314 345 418 425 46 446 454 351 271 163 136 535"   # 06uc7cis k5 s43
  "333 178 366 425 287 329 260 473 34 47 496 158 255 76 495 465 404 179 42 206"   # e35ds248 k5 s44
  "351 191 30 97 235 187 52 533 12 104 374 388 66 383 299 128 174 206 233 443"   # 0r5j0p8y k6 s42
  "4 260 507 34 141 97 391 376 453 5 492 298 83 383 437 94 357 333 281 47"   # osjfnxz0 k6 s43
  "34 107 358 58 282 195 516 533 218 261 106 158 219 221 296 297 321 402 464 475"   # w8sbas7o k6 s44
  "314 88 273 49 204 263 228 357 22 515 383 451 182 298 355 391 41 121 290 296"   # cgq46nnk k2 s42
  "468 452 357 487 317 329 363 298 65 391 451 472 228 53 393 386 437 300 314 49"   # c8yihf5o k2 s43
  "531 315 69 479 5 170 366 108 376 391 281 345 100 107 169 533 287 278 395 454"   # yw4mao1d k2 s44
  "520 479 289 384 320 78 95 287 400 35 317 271 449 504 507 237 355 379 107 383"   # x4cskyot k3 s42
  "261 70 496 78 67 247 95 284 57 32 283 366 503 245 487 278 431 237 128 62"   # 1eqwuq6m k3 s44
  "95 136 195 237 224 78 298 471 363 328 289 329 284 523 507 66 400 65 287 379"   # m4ibxlhu k4 s42
  "252 84 161 218 407 22 384 445 526 455 123 298 535 10 66 355 393 209 468 451"   # yu0vk259 k4 s43
  "393 270 376 435 271 40 100 286 496 191 416 121 232 315 220 303 58 473 46 487"   # 7s0p1b8q k4 s44
  "1 58 123 152 317 345 430 451 46 252 107 218 136 496 10 391 535 72 418 471"   # hzttkj82 k5 s42
  "298 317 345 479 487 128 300 357 451 76 303 388 449 452 136 218 58 96 161 391"   # fkufm4ol k5 s43
  "383 312 30 317 329 435 308 382 495 34 445 345 451 287 10 337 244 363 493 460"   # 071oqqu5 k5 s44
  "100 58 393 96 449 435 104 215 230 328 191 178 238 333 391 535 174 250 320 337"   # p4ltvhdf k6 s42
  "218 351 83 189 468 121 354 479 504 258 492 533 85 309 507 66 178 355 169 391"   # 9ti3ocxm k6 s43
  "220 83 311 17 435 85 100 79 526 96 384 437 393 182 270 496 487 181 152 337"   # ly30vz8m k6 s44
)

read -r WID K SEED ITER PARTNER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MAP_IDS="${MAPIDS[$SLURM_ARRAY_TASK_ID]}"
N_STEPS=$((K * 201))
ITER6=$(printf "%06d" "$ITER")
OUT="outputs/videos_grid/${WID}"
echo "[rendergrid] task=$SLURM_ARRAY_TASK_ID partner=$PARTNER wid=$WID k=$K seed=$SEED iter=$ITER n_steps=$N_STEPS"
echo "[rendergrid] map_ids: $MAP_IDS"

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
