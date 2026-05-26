#!/bin/bash
# Local driver for offline k=4 gb=3 eval across 35 wids, with bounded
# concurrency. Each worker calls eval_k4_gb3_one.py for one wid (loops 8
# checkpoints internally, logs offline_eval/* to the original wandb run).
#
# Concurrency cap is essential — this session has SLURM_MEM_PER_NODE=24 GB,
# each puffer-eval subprocess uses ~8 GB RSS (nuplan_hard map data), so >2
# parallel evals trigger the cgroup OOM-killer.
#
# Skip-if-done semantics: if logs/offline_eval/<wid>.done exists, skip. So
# ctrl-c + rerun is safe; a separate SLURM submission can also touch the
# .done marker to coordinate (same shared filesystem).
#
# Usage:
#   bash scripts/adaptive/eval_k4_gb3_all_local.sh                 # defaults: 2 concurrent, 20 rollouts
#   bash scripts/adaptive/eval_k4_gb3_all_local.sh 2 20            # explicit
#   WIDS_FILTER="rwg5a65x icmjygwf" bash scripts/.../all_local.sh

set -u

CONCURRENCY=${1:-2}
ROLLOUTS=${2:-20}
NUM_MAPS=${NUM_MAPS:-540}
NUM_AGENTS=${NUM_AGENTS:-540}
LOG_DIR=logs/offline_eval
mkdir -p "$LOG_DIR"

ALL_WIDS=(
  rwg5a65x icmjygwf 6rv8gcrr
  ipdv2oag nbipb5q9 l9wv41ct
  hke6hyik 3s24do45
  rx3yj0k7 diocrfd9 se7ovksg
  uljixs7j 251vz655 0rkojso4
  rmsghbiu h7fajqan wplaas1l
  5obko2iy umaskfka a37ay3nb
  4cowebjw t1bkn7fq s1ro6tzv
  mpyo1ucm auspoa8z 4tqk602k
  38g805cy he3wmzo4 0da431f2
  o13lmh0q 5fhn4zng 438s7mb2
  96f15g3o 6nrtfrex huk9yuqd
)

if [ -n "${WIDS_FILTER:-}" ]; then
  read -r -a WIDS <<< "$WIDS_FILTER"
else
  WIDS=("${ALL_WIDS[@]}")
fi

echo "[driver] ${#WIDS[@]} wids, $CONCURRENCY concurrent, $ROLLOUTS rollouts × $NUM_MAPS maps × $NUM_AGENTS agents"
echo "[driver] start: $(date)"
TOTAL_T0=$(date +%s)

run_one_wid() {
  local wid=$1
  local done_marker="$LOG_DIR/${wid}.done"
  local log_file="$LOG_DIR/${wid}.log"
  if [ -f "$done_marker" ]; then
    echo "[driver] SKIP $wid (done)"
    return 0
  fi
  local t0=$(date +%s)
  echo "[driver] START $wid  $(date '+%H:%M:%S')"
  : > "$log_file"
  if xvfb-run -a python scripts/adaptive/eval_k4_gb3_one.py \
       --wid "$wid" \
       --num-rollouts "$ROLLOUTS" \
       --num-maps "$NUM_MAPS" \
       --num-agents "$NUM_AGENTS" >> "$log_file" 2>&1; then
    local elapsed=$(( $(date +%s) - t0 ))
    echo "[driver] OK    $wid  ${elapsed}s"
    touch "$done_marker"
  else
    local ec=$?
    local elapsed=$(( $(date +%s) - t0 ))
    echo "[driver] FAIL  $wid  ec=$ec  ${elapsed}s  (see $log_file)"
  fi
}
export -f run_one_wid
export LOG_DIR ROLLOUTS NUM_MAPS NUM_AGENTS

# Use xargs -P for bounded-concurrency pool. Each wid is one xargs input;
# xargs spawns at most CONCURRENCY parallel processes.
printf '%s\n' "${WIDS[@]}" \
  | xargs -P "$CONCURRENCY" -I{} bash -c 'run_one_wid "$@"' _ {}

echo "[driver] total: $(( $(date +%s) - TOTAL_T0 ))s"
echo "[driver] done markers: $(ls $LOG_DIR/*.done 2>/dev/null | wc -l) / ${#WIDS[@]}"
