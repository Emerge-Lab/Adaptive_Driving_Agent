#!/bin/bash
# Offline eval wrapper for the conditional recovery metric.
#
# Usage:
#   bash scripts/eval_recovery.sh <wandb_id> [<gpu>] [<num_rollouts>] [<num_maps>]
#
# Defaults:
#   gpu=0
#   num_rollouts=500
#   num_maps=500
#
# Looks up the latest model_*.pt for the given wandb_id under
# experiments/puffer_adaptive_drive_<wid>/, runs `puffer eval` with
# human_replay enabled, and surfaces the recovery/* metrics in stdout.
#
# Doesn't change wandb history of the original run — eval logs go to a
# fresh ada_recovery_eval project so they're easy to find.

set -e

WID=${1:?"need wandb_id"}
GPU=${2:-0}
NROLL=${3:-500}
NMAPS=${4:-500}
# CACHE_RESET=1 (env var) propagates as RECOVERY_CACHE_RESET_PER_SCENARIO
# to the puffer eval subprocess; the evaluator reads that env var and
# resets state at scenario boundaries.
export RECOVERY_CACHE_RESET_PER_SCENARIO=${CACHE_RESET:-0}

CKPT_DIR=/workspace/ADA/experiments/puffer_adaptive_drive_${WID}
CKPT=$(ls ${CKPT_DIR}/model_puffer_adaptive_drive_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint found for wid=$WID under $CKPT_DIR" >&2
  exit 1
fi
echo "Using checkpoint: $(basename $CKPT)"

cd /workspace/ADA
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# We use puffer's eval mode (separate from training). It reads the
# checkpoint, runs HumanReplayEvaluator.rollout, and prints the metric
# dict as the last block of stdout (between HUMAN_REPLAY_METRICS_START/END).
exec xvfb-run -a stdbuf -oL -eL puffer eval puffer_adaptive_drive \
  --load-model-path $CKPT \
  --eval.wosac-realism-eval False \
  --eval.human-replay-eval True \
  --eval.human-replay-num-agents 100 \
  --eval.human-replay-num-maps $NMAPS \
  --eval.human-replay-num-rollouts $NROLL \
  --eval.human-replay-control-mode control_vehicles \
  --eval.map-dir resources/drive/binaries/nuplan \
  --eval.num-maps 20 \
  --env.k-scenarios 3 \
  --env.scenario-length 91 \
  --train.horizon 273 \
  --env.goal-behavior 2 \
  --env.conditioning.type none \
  --env.conditioning.collision-weight-lb -1.0 --env.conditioning.collision-weight-ub 0.0 \
  --env.conditioning.offroad-weight-lb -0.4 --env.conditioning.offroad-weight-ub 0.0 \
  --env.conditioning.goal-weight-lb 0.0 --env.conditioning.goal-weight-ub 1.0 \
  --env.conditioning.entropy-weight-lb 0.0 --env.conditioning.entropy-weight-ub 0.001 \
  --env.conditioning.discount-weight-lb 0.8 --env.conditioning.discount-weight-ub 0.98
