#!/bin/bash
# Train every co-player variant we currently care about, in series.
#
# Use this to eyeball that learning is happening across the matrix
# (we can't see learning curves from CI, only from running and watching wandb).
# Once these are trained, feed the resulting checkpoints into
# scripts/adaptive/* to verify adaptive-agent training works end-to-end.
#
# Usage:
#   bash scripts/ablations/all_coplayers.sh                       # full matrix
#   bash scripts/ablations/all_coplayers.sh --quick               # tiny budgets, smoke-style
#   bash scripts/ablations/all_coplayers.sh --datasets nuplan     # one dataset
#   bash scripts/ablations/all_coplayers.sh --archs Recurrent     # one architecture
#
# Artifacts: wandb run ids → experiments/puffer_drive_<run>.pt

set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

DATASETS=(womd nuplan)
ARCHS=(Recurrent Transformer)
COND_TYPES=(none all)
QUICK=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --datasets) IFS=',' read -ra DATASETS <<< "$2"; shift 2 ;;
    --archs)    IFS=',' read -ra ARCHS    <<< "$2"; shift 2 ;;
    --cond)     IFS=',' read -ra COND_TYPES <<< "$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *)  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  TIMESTEPS=200000
  CHECKPOINT_INTERVAL=5
  NUM_MAPS_NUPLAN=20
  NUM_MAPS_WOMD=20
else
  TIMESTEPS=500000000
  CHECKPOINT_INTERVAL=50
  NUM_MAPS_NUPLAN=5000
  NUM_MAPS_WOMD=10000
fi

map_dir_for() {
  case "$1" in
    nuplan) echo "resources/drive/binaries/nuplan" ;;
    womd)   echo "resources/drive/binaries/training" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

num_maps_for() {
  case "$1" in
    nuplan) echo "$NUM_MAPS_NUPLAN" ;;
    womd)   echo "$NUM_MAPS_WOMD" ;;
  esac
}

cond_args_for() {
  # All sweeps fix entropy_lb=0, discount_ub=1; vary entropy_ub and discount_lb.
  case "$1" in
    none) echo "--env.conditioning.type none" ;;
    all)  echo "--env.conditioning.type all \
                --env.conditioning.entropy-weight-lb 0 \
                --env.conditioning.entropy-weight-ub 0.1 \
                --env.conditioning.discount-weight-lb 0.8 \
                --env.conditioning.discount-weight-ub 1.0" ;;
    *) echo "unknown conditioning $1" >&2; exit 1 ;;
  esac
}

run_one() {
  local dataset="$1"
  local arch="$2"
  local cond="$3"
  local map_dir num_maps cond_args tag
  map_dir=$(map_dir_for "$dataset")
  num_maps=$(num_maps_for "$dataset")
  cond_args=$(cond_args_for "$cond")
  tag="coplayer_${dataset}_${arch,,}_cond-${cond}"

  echo
  echo "=== $tag ==="
  echo "    map_dir=$map_dir num_maps=$num_maps timesteps=$TIMESTEPS"
  echo

  # shellcheck disable=SC2086
  puffer train puffer_drive --wandb \
    --wandb-project ada_coplayer_ablation \
    --tag "$tag" \
    --policy-architecture "$arch" \
    --rnn-name "$arch" \
    --env.map-dir "$map_dir" \
    --env.num-maps "$num_maps" \
    --eval.map-dir "$map_dir" \
    --train.total-timesteps "$TIMESTEPS" \
    --train.checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --train.render False \
    $cond_args \
    "${EXTRA_ARGS[@]}"
}

for dataset in "${DATASETS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for cond in "${COND_TYPES[@]}"; do
      run_one "$dataset" "$arch" "$cond"
    done
  done
done

echo
echo "Done. Checkpoints in experiments/puffer_drive_*.pt; wandb project: ada_coplayer_ablation"
