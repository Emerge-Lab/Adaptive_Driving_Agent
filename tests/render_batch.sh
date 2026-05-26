#!/bin/bash
# Batch driver: probe → render N random env_ids per map_seed → heatmaps.
#
# Per seed: 1 probe (~15s, captures attention for all 8 agents) + K renders
# (~150s each) + K heatmap plots (~5s each).
#
# Output layout (under --out-root):
#   batch/
#     seed{N}/
#       attn_layer0.npz, garbage_mask.npz, active.npz, summary.txt  (shared)
#       epoch_000000_human_replay_env{E}_map{M}.mp4                 (per env)
#       attention_over_time_env{E}.png                              (per env)
#       attention_per_head_env{E}.png                               (per env)
#
# Usage:
#   bash tests/render_batch.sh <ckpt> <info> <out_root> [seeds] [envs_per_seed]
#
# Defaults: seeds="42 43 44 45 46"  envs_per_seed=2  ⇒ 10 pairs ≈ 26 min on L40S

set -u

CKPT=${1:?need ckpt}
INFO=${2:?need info.json}
OUT_ROOT=${3:?need out root (e.g. outputs/inspect/rwg5a65x_iter76/batch)}
SEEDS=${4:-"42 43 44 45 46"}
ENVS_PER_SEED=${5:-2}
MAP_DIR=${MAP_DIR:-resources/drive/binaries/nuplan_hard}
NUM_AGENTS=${NUM_AGENTS:-8}
N_STEPS=${N_STEPS:-804}
K=${K:-4}
SCEN_LEN=${SCEN_LEN:-201}

mkdir -p "$OUT_ROOT"
echo "[batch] ckpt=$CKPT  out=$OUT_ROOT  seeds=[$SEEDS]  envs_per_seed=$ENVS_PER_SEED  map_dir=$MAP_DIR"

TOTAL_T0=$(date +%s)
for SEED in $SEEDS; do
  SEED_DIR="$OUT_ROOT/seed$SEED"
  mkdir -p "$SEED_DIR"
  echo ""
  echo "=========== map_seed=$SEED ==========="

  # 1. Probe pass: captures attention for all $NUM_AGENTS agents, no render.
  echo "[batch] probe (no render)..."
  T0=$(date +%s)
  xvfb-run -a python tests/render_all_scenes.py \
    --checkpoint "$CKPT" --info "$INFO" \
    --num-agents "$NUM_AGENTS" --n-steps "$N_STEPS" \
    --map-seed "$SEED" --map-dir "$MAP_DIR" \
    --no-render --out "$SEED_DIR" > "$SEED_DIR/probe.log" 2>&1
  echo "[batch]   probe done in $(( $(date +%s) - T0 ))s"
  tail -12 "$SEED_DIR/probe.log"

  # 2. Pick $ENVS_PER_SEED random env_ids from [0, NUM_AGENTS).
  ENVS=$(python3 -c "import random; random.seed($SEED); print(' '.join(str(x) for x in random.sample(range($NUM_AGENTS), $ENVS_PER_SEED)))")
  echo "[batch] random env_ids for seed=$SEED: $ENVS"

  # 3. Render each selected env_id.
  for ENV_ID in $ENVS; do
    echo "[batch] render env_id=$ENV_ID..."
    T0=$(date +%s)
    xvfb-run -a python tests/render_one_scene.py \
      --checkpoint "$CKPT" --info "$INFO" \
      --env-id "$ENV_ID" --n-steps "$N_STEPS" \
      --num-agents "$NUM_AGENTS" \
      --map-seed "$SEED" --map-dir "$MAP_DIR" \
      --out "$SEED_DIR" > "$SEED_DIR/render_env${ENV_ID}.log" 2>&1
    echo "[batch]   render env=$ENV_ID done in $(( $(date +%s) - T0 ))s"

    # 4. Heatmap for that agent (from the probe's attention data).
    python tests/plot_attention.py \
      --root "$SEED_DIR" --flat \
      --agent "$ENV_ID" --k "$K" --scen-len "$SCEN_LEN" \
      --modes human_replay > "$SEED_DIR/plot_env${ENV_ID}.log" 2>&1
    echo "[batch]   heatmap env=$ENV_ID done"
  done
done

echo ""
echo "[batch] total wall: $(( $(date +%s) - TOTAL_T0 ))s"
echo "[batch] outputs under $OUT_ROOT"
ls "$OUT_ROOT" | head -20
