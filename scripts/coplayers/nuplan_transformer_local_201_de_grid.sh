#!/bin/bash
set -e

# Co-player training: 2D sweep over (discount_lb, entropy_ub).
# Grid: 3 discount_lb × 5 entropy_ub = 15 partners. Run 5 in parallel,
# 3 sequential batches, ~4.5h per batch, ~13.5h total wall-clock.
#
# Why this sweep: extends the 5-partner entropy sweep with discount_lb
# variation. Each (d, e) pair gives a partner with a distinct
# behavior profile, used to study which partner properties drive
# adaptation gain on the ada_delta plot.
#
# Defaults to GPUs 0-4 (override: GPUS="..." bash <script>).
# Stop everything: tmux kill-session -t coplayer_de_grid

GPUS=${GPUS:-"0 1 2 3 4"}
SESSION=coplayer_de_grid
DRIVER=/workspace/ADA/scripts/coplayers/_de_grid_driver.sh

# Driver script is in a sibling file (._de_grid_driver.sh). Sanity-check it exists.
if [ ! -x "$DRIVER" ]; then
  echo "ERROR: driver script missing or not executable: $DRIVER" >&2
  exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n driver
tmux send-keys -t "$SESSION:driver" "GPUS='$GPUS' bash $DRIVER" C-m

echo "Launched. tmux session: $SESSION"
echo "Driver log: /tmp/coplayer_de_grid_driver.log"
echo "Per-job logs: /tmp/coplayer_de_grid_d<DLB>_e<EUB>.log"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
