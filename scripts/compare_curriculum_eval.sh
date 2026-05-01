#!/bin/bash
# Sequential offline eval of curr vs nocurr e=0.5 curriculum policies.
#
# Runs puffer eval (human_replay) on 300 held-out scenes for each
# checkpoint, captures HUMAN_REPLAY_METRICS_* JSON, computes:
#   - per-scenario success rate
#   - P(s1=success | s0=success)
#   - P(s1=success | s0=fail)         <- the recovery / adaptation metric
#   - ada_delta_score = mean(s1) - mean(s0)
#
# Sequential (not parallel) so we don't OOM the running k_eff jobs that
# share the host. Each eval ~25-40 GB RAM; the running k_eff at horizon=804
# already eats ~370 GB so we have ~130 GB headroom.
#
# Usage:
#   bash scripts/compare_curriculum_eval.sh [GPU]
# Default GPU: 6 (0-3 idle but stale CUDA contexts; 4-5 in use by k_eff;
# 6-7 clean).

set -e

GPU=${1:-6}
NMAPS=300
NROLL=1                         # 300 unique scenes per rollout = 300 (s0,s1) pairs
NAGENTS=$NMAPS                  # one SDC per scene
MAP_DIR=resources/drive/binaries/nuplan_201_heldout300

CURR_WID=hprfn8dc               # curr_e0.5
NOCURR_WID=7wm1sk5v             # nocurr_e0.5
OUT_DIR=/tmp/curriculum_compare_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUT_DIR
echo "Outputs: $OUT_DIR"

cd /workspace/ADA
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

run_eval() {
  local LABEL=$1
  local WID=$2
  local CKPT_DIR=/workspace/ADA/experiments/puffer_adaptive_drive_${WID}
  local CKPT=$(ls ${CKPT_DIR}/model_puffer_adaptive_drive_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint for $WID" >&2
    return 1
  fi
  local OUT_LOG=$OUT_DIR/${LABEL}_${WID}.log
  local OUT_JSON=$OUT_DIR/${LABEL}_${WID}.json
  echo "[$LABEL] checkpoint: $(basename $CKPT)"
  echo "[$LABEL] starting eval → $OUT_LOG"

  xvfb-run -a stdbuf -oL -eL puffer eval puffer_adaptive_drive \
    --load-model-path $CKPT \
    --eval.wosac-realism-eval False \
    --eval.human-replay-eval True \
    --eval.human-replay-num-agents $NAGENTS \
    --eval.human-replay-num-maps $NMAPS \
    --eval.human-replay-num-rollouts $NROLL \
    --eval.human-replay-control-mode control_vehicles \
    --eval.map-dir $MAP_DIR \
    --eval.num-maps 20 \
    --env.k-scenarios 2 \
    --env.scenario-length 201 \
    --train.horizon 402 \
    --env.goal-behavior 2 \
    --env.conditioning.type none 2>&1 | tee $OUT_LOG

  # Extract the JSON between markers.
  awk '/HUMAN_REPLAY_METRICS_START/{flag=1; next} /HUMAN_REPLAY_METRICS_END/{flag=0} flag' $OUT_LOG > $OUT_JSON
  if [ ! -s "$OUT_JSON" ]; then
    echo "[$LABEL] WARNING: no metrics JSON captured" >&2
    return 1
  fi
  echo "[$LABEL] metrics → $OUT_JSON ($(wc -c <$OUT_JSON) bytes)"
}

# --- run sequentially ---
run_eval "nocurr_e05" "$NOCURR_WID"
run_eval "curr_e05"   "$CURR_WID"

# --- compute conditional recovery metrics ---
python3 - <<PYEOF
import json, glob, os, sys
out_dir = "$OUT_DIR"
results = {}
for label in ("nocurr_e05", "curr_e05"):
    p = glob.glob(f"{out_dir}/{label}_*.json")[0]
    d = json.load(open(p))
    log = d.get("per_agent_success_log") or []
    s0 = [r["s0"] for r in log]
    s1 = [r["s1"] for r in log]
    n = len(log)
    n_s0 = sum(s0)
    n_s1 = sum(s1)
    n_s0_pass_s1_pass = sum(1 for r in log if r["s0"] and r["s1"])
    n_s0_fail = sum(1 for r in log if not r["s0"])
    n_s0_fail_s1_pass = sum(1 for r in log if not r["s0"] and r["s1"])
    n_s0_pass = sum(1 for r in log if r["s0"])
    p_s1_given_s0_pass = n_s0_pass_s1_pass / n_s0_pass if n_s0_pass else float("nan")
    p_s1_given_s0_fail = n_s0_fail_s1_pass / n_s0_fail if n_s0_fail else float("nan")
    results[label] = dict(
        n=n,
        s0_rate=n_s0 / n,
        s1_rate=n_s1 / n,
        ada_delta=(n_s1 - n_s0) / n,
        p_s1_given_s0_pass=p_s1_given_s0_pass,
        p_s1_given_s0_fail=p_s1_given_s0_fail,
        n_s0_pass=n_s0_pass,
        n_s0_fail=n_s0_fail,
    )

def fmt(v):
    if v != v: return "  nan "
    return f"{v:6.3f}"

print()
print("=" * 72)
print(f"{'metric':35s}  {'nocurr_e05':>12s}  {'curr_e05':>12s}  {'Δ':>8s}")
print("-" * 72)
for k in ("n", "s0_rate", "s1_rate", "ada_delta",
          "p_s1_given_s0_pass", "p_s1_given_s0_fail",
          "n_s0_pass", "n_s0_fail"):
    a = results["nocurr_e05"][k]
    b = results["curr_e05"][k]
    if isinstance(a, int) and isinstance(b, int):
        delta = b - a
        print(f"{k:35s}  {a:>12d}  {b:>12d}  {delta:>+8d}")
    else:
        delta = (b - a) if (a == a and b == b) else float("nan")
        print(f"{k:35s}  {fmt(a):>12s}  {fmt(b):>12s}  {fmt(delta):>+8s}")
print("=" * 72)
print()
print("Reading: ada_delta is the headline (positive = adaptation).")
print("  p_s1_given_s0_fail is the recovery rate — how often the policy")
print("  recovers in s1 after failing in s0. The curriculum hypothesis is")
print("  this should be HIGHER for curr than nocurr.")
print()
print(f"Outputs: {out_dir}/")

with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(results, f, indent=2)
PYEOF
