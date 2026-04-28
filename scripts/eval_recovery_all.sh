#!/bin/bash
# Run conditional-recovery eval on all 4 k=3 adaptive checkpoints in parallel
# (one per GPU). Writes each run's stdout to /tmp/recovery_<wid>.log and the
# parsed JSON metrics to /tmp/recovery_<wid>.json.
#
# Usage:
#   bash scripts/eval_recovery_all.sh [<num_rollouts>] [<num_maps>]
#
# Defaults: num_rollouts=300, num_maps=300 (smaller than prod 500 for speed)

set -e

NROLL=${1:-300}
NMAPS=${2:-300}

# (wid, gpu) pairs — one per k=3 adaptive resume run.
RUNS=(
  "y9tges7d 4"
  "bzzosxsg 5"
  "5p6tl3pt 6"
  "3p5ome2t 7"
)

PIDS=()
for entry in "${RUNS[@]}"; do
  read -r WID GPU <<< "$entry"
  echo "Launching eval: wid=$WID gpu=$GPU rollouts=$NROLL maps=$NMAPS"
  bash /workspace/ADA/scripts/eval_recovery.sh "$WID" "$GPU" "$NROLL" "$NMAPS" \
    > "/tmp/recovery_${WID}.log" 2>&1 &
  PIDS+=($!)
done

echo
echo "Started ${#PIDS[@]} eval procs (PIDs: ${PIDS[*]})"
echo "Waiting for all to complete (each ~5-10 min at $NROLL rollouts)..."
echo

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "ERROR: pid=$pid failed (see logs)"
    FAILED=$((FAILED+1))
  fi
done
echo
echo "All eval procs done. Failed: $FAILED/${#PIDS[@]}"

# Extract HUMAN_REPLAY_METRICS_START..END JSON block from each log into a json file
for entry in "${RUNS[@]}"; do
  read -r WID _ <<< "$entry"
  awk '/HUMAN_REPLAY_METRICS_START/{flag=1;next}/HUMAN_REPLAY_METRICS_END/{flag=0}flag' \
    /tmp/recovery_${WID}.log > /tmp/recovery_${WID}.json || true
  if [ -s /tmp/recovery_${WID}.json ]; then
    echo "  ${WID}: metrics extracted to /tmp/recovery_${WID}.json"
  else
    echo "  ${WID}: NO METRICS BLOCK FOUND in /tmp/recovery_${WID}.log (eval likely crashed)"
  fi
done
