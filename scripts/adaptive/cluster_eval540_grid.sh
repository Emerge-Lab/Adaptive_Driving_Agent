#!/bin/bash
#SBATCH --job-name=eval540grid
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
# Task #30: re-eval the NEW co-player x k grid final checkpoints at 540 maps x 20
# rollouts (human-replay), generalized over k. Runs on l40s/priority so it does
# NOT compete with the h100 training resumes (jobs 11287601, 11131115).
#
# This batch = the 39 checkpoints that are DONE. The 6 in-flight cells
# (resuming: bhx6zxn0 052n7brp cei22yc1 citbzhdc blyerjec; rerun: 9qewt905)
# get a follow-up batch once they reach target.
#
# Output CSVs -> outputs/eval540_grid/ (kept separate from the old 15-run
# outputs/eval540/ because the analysis globs on (k,seed) only and would
# otherwise clobber across the 3 partners that share each k/seed). wid in the
# filename joins back to partner via scripts/adaptive/final_runs_manifest.csv.
#
# Config = "wid k seed iter partner" (partner is for the log line only).
CONFIGS=(
  # miku2puk  e_ub=0.05
  "6opvas42 2 42 228 miku2puk"
  "3dlsmo4v 2 43 228 miku2puk"
  "6gzqx4gj 2 44 228 miku2puk"
  "io7kp2cq 3 42 152 miku2puk"
  "xr137mjs 3 43 152 miku2puk"
  "rf4p3hy6 3 44 152 miku2puk"
  "9gc19bcy 4 42 114 miku2puk"
  "246wih6m 5 42 365 miku2puk"
  "w1u39swd 5 43 365 miku2puk"
  "f754lim2 5 44 365 miku2puk"
  "ijwa3y93 6 42 304 miku2puk"
  "hk5gtm99 6 43 304 miku2puk"
  "yl6tfvb0 6 44 304 miku2puk"
  # m2ygolog  e_ub=0.20
  "obrwxqqy 2 42 228 m2ygolog"
  "gesyc4j9 2 43 228 m2ygolog"
  "s790xh0d 2 44 228 m2ygolog"
  "n46bkreg 3 43 152 m2ygolog"
  "ftxa55g3 4 42 114 m2ygolog"
  "c0k9uqhc 4 44 114 m2ygolog"
  "72dilduo 5 42 365 m2ygolog"
  "06uc7cis 5 43 365 m2ygolog"
  "e35ds248 5 44 365 m2ygolog"
  "0r5j0p8y 6 42 304 m2ygolog"
  "osjfnxz0 6 43 304 m2ygolog"
  "w8sbas7o 6 44 304 m2ygolog"
  # 6rauydj2  e_ub=0.50
  "cgq46nnk 2 42 228 6rauydj2"
  "c8yihf5o 2 43 220 6rauydj2"
  "yw4mao1d 2 44 228 6rauydj2"
  "x4cskyot 3 42 152 6rauydj2"
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
# Submit: sbatch scripts/adaptive/cluster_eval540_grid.sh

read -r WID K SEED ITER PARTNER <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[eval540grid] task=$SLURM_ARRAY_TASK_ID partner=$PARTNER wid=$WID k=$K seed=$SEED iter=$ITER"

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
