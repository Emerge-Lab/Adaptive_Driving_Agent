#!/bin/bash
# Run the CONTROL variant: K/V cache reset at every scenario boundary.
# Same checkpoints/maps/rollouts as eval_recovery_all.sh, on the same 4 GPUs.
# Output goes to /tmp/recovery_control_<wid>.json (note the _control suffix).
set -e

NROLL=${1:-50}
NMAPS=${2:-100}

RUNS=(
  "y9tges7d 4"
  "bzzosxsg 5"
  "5p6tl3pt 6"
  "3p5ome2t 7"
)

PIDS=()
for entry in "${RUNS[@]}"; do
  read -r WID GPU <<< "$entry"
  echo "Launching CONTROL eval: wid=$WID gpu=$GPU rollouts=$NROLL maps=$NMAPS"
  CACHE_RESET=1 bash /workspace/ADA/scripts/eval_recovery.sh "$WID" "$GPU" "$NROLL" "$NMAPS" \
    > "/tmp/recovery_control_${WID}.log" 2>&1 &
  PIDS+=($!)
done

echo "Started ${#PIDS[@]} control eval procs"
echo "Waiting..."
FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=$((FAILED+1))
done
echo "Done. Failed: $FAILED/${#PIDS[@]}"

for entry in "${RUNS[@]}"; do
  read -r WID _ <<< "$entry"
  awk '/HUMAN_REPLAY_METRICS_START/{flag=1;next}/HUMAN_REPLAY_METRICS_END/{flag=0}flag' \
    /tmp/recovery_control_${WID}.log > /tmp/recovery_control_${WID}.json || true
  size=$(wc -c < /tmp/recovery_control_${WID}.json 2>/dev/null)
  echo "  $WID: $size bytes -> /tmp/recovery_control_${WID}.json"
done
