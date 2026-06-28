#!/bin/bash
#SBATCH --job-name=eval540_R
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-59
#
# RETURN-based re-eval over all 60 grid cells. Same checkpoints as the original
# eval540 success-only pass, but the evaluator now also logs per-trial SUMMED
# REWARD (return) per (rollout, map, trial). eval_final_540.py passes the
# TRAINING reward weights (lane 0.05, collision/offroad −0.5) so the eval-time
# return is comparable to what the agent optimized; reward weights do NOT
# affect rollouts or success bits, only the logged scalar.
#
# Outputs -> outputs/eval540_return/
#   per_map_k{k}_seed{s}_{wid}.csv     (success — same schema as the old eval)
#   per_map_R_k{k}_seed{s}_{wid}.csv   (return  — the new continuous metric)
#
# Cells (60 total) = 39 + 6 + 15 from cluster_eval540_grid.sh,
# cluster_eval540_grid_batch2.sh, cluster_eval540_all.sh respectively.
# Format: "wid k seed iter partner"

CONFIGS=(
  # ----- miku2puk  e_ub=0.05  (13) -----
  "6opvas42 2 42 228 miku2puk"
  "3dlsmo4v 2 43 228 miku2puk"
  "6gzqx4gj 2 44 228 miku2puk"
  "io7kp2cq 3 42 152 miku2puk"
  "xr137mjs 3 43 152 miku2puk"
  "rf4p3hy6 3 44 152 miku2puk"
  "9gc19bcy 4 42 114 miku2puk"
  "bhx6zxn0 4 43 114 miku2puk"
  "9qewt905 4 44 114 miku2puk"
  "246wih6m 5 42 365 miku2puk"
  "w1u39swd 5 43 365 miku2puk"
  "f754lim2 5 44 365 miku2puk"
  "ijwa3y93 6 42 304 miku2puk"
  "hk5gtm99 6 43 304 miku2puk"
  "yl6tfvb0 6 44 304 miku2puk"

  # ----- 2e029h15  e_ub=0.10  (15) -----
  "ofm0rrbm 2 42 228 2e029h15"
  "r2wr75ay 2 43 170 2e029h15"
  "vx2g4pcg 2 44 140 2e029h15"
  "0qsa6hku 3 42 152 2e029h15"
  "lkgv9a0b 3 43 152 2e029h15"
  "pbvf72ym 3 44 100 2e029h15"
  "qxw6c0jh 4 42 110 2e029h15"
  "ufmegw4l 4 43 114 2e029h15"
  "jsckmpha 4 44  80 2e029h15"
  "jc264zfr 5 42 365 2e029h15"
  "nlzthr49 5 43 365 2e029h15"
  "i09bkafr 5 44 365 2e029h15"
  "f3p7nms8 6 42 304 2e029h15"
  "r5dgbtmg 6 43 304 2e029h15"
  "macfatw8 6 44 304 2e029h15"

  # ----- m2ygolog  e_ub=0.20  (15) -----
  "obrwxqqy 2 42 228 m2ygolog"
  "gesyc4j9 2 43 228 m2ygolog"
  "s790xh0d 2 44 228 m2ygolog"
  "052n7brp 3 42 152 m2ygolog"
  "n46bkreg 3 43 152 m2ygolog"
  "cei22yc1 3 44 152 m2ygolog"
  "ftxa55g3 4 42 114 m2ygolog"
  "citbzhdc 4 43 114 m2ygolog"
  "c0k9uqhc 4 44 114 m2ygolog"
  "72dilduo 5 42 365 m2ygolog"
  "06uc7cis 5 43 365 m2ygolog"
  "e35ds248 5 44 365 m2ygolog"
  "0r5j0p8y 6 42 304 m2ygolog"
  "osjfnxz0 6 43 304 m2ygolog"
  "w8sbas7o 6 44 304 m2ygolog"

  # ----- 6rauydj2  e_ub=0.50  (15) -----
  "cgq46nnk 2 42 228 6rauydj2"
  "c8yihf5o 2 43 220 6rauydj2"
  "yw4mao1d 2 44 228 6rauydj2"
  "x4cskyot 3 42 152 6rauydj2"
  "blyerjec 3 43 152 6rauydj2"
  "1eqwuq6m 3 44 152 6rauydj2"
  "m4ibxlhu 4 42 114 6rauydj2"
  "yu0vk259 4 43 114 6rauydj2"
  "7s0p1b8q 4 44 114 6rauydj2"
  "hzttkj82 5 42 365 6rauydj2"
  "fkufm4ol 5 43 365 6rauydj2"
  "071oqqu5 5 44 365 6rauydj2"
  "p4ltvhdf 6 42 304 6rauydj2"
  "9ti3ocxm 6 43 304 6rauydj2"
  "ly30vz8m 6 44 304 6rauydj2"
)
# Submit: sbatch scripts/adaptive/cluster_eval540_return.sh

read -r WID K SEED ITER PARTNER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540_R] task=$SLURM_ARRAY_TASK_ID partner=$PARTNER wid=$WID k=$K seed=$SEED iter=$ITER"

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
     --out-dir outputs/eval540_return \
     --return-dir outputs/eval540_return \
     --table-prefix eval540_R_20r \
     --timeout-sec 18000
 "
