#!/bin/bash
# Run-the-whole-pipeline smoke test.
#
# In ~3 minutes, this:
#   1. trains a tiny LSTM co-player (~50k steps)
#   2. trains a tiny adaptive ego against it (~50k steps, k=2)
#   3. renders all four modes (baseline, coplayer, adaptive vs coplayer, human-replay)
#   4. asserts each render produced its mp4 in <run_dir>/renders/
#   5. runs the pytest suite for unit-test coverage
#
# Useful for: "did my last refactor break the integration?"
#
# Numbers expected: with normal conditioning we eventually hit ~0.3 score
# at ~50M steps, so this smoke is *structural* (does it run?) not
# performance-checking. The score will be near zero — that's expected.
#
# Usage:
#   bash scripts/smoke.sh
#
# The script returns non-zero on any failure so it composes with `set -e`
# in CI / pre-push hooks.

set -eo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

FIXTURE_MAPS="tests/fixtures/maps"
# Use a unique smoke dir per run so a previous run holding NFS file
# handles can't block ours.
SMOKE_DIR="experiments/smoke_$$"
COPLAYER_DATA_DIR="$SMOKE_DIR/coplayer"
ADAPTIVE_DATA_DIR="$SMOKE_DIR/adaptive"
mkdir -p "$COPLAYER_DATA_DIR" "$ADAPTIVE_DATA_DIR"

echo "==> 1/5: building C extension (incremental)"
BUILD_LOG="$SMOKE_DIR/build.log"
python setup.py build_ext --inplace >"$BUILD_LOG" 2>&1 || {
  echo "FAIL: C build failed"; tail -40 "$BUILD_LOG"; exit 1;
}
tail -3 "$BUILD_LOG"

echo
echo "==> 2/5: training tiny co-player (~50k steps, LSTM, conditioning=all)"
COPLAYER_LOG="$SMOKE_DIR/coplayer.log"
puffer train puffer_drive \
  --policy-architecture Recurrent \
  --rnn-name Recurrent \
  --train.data-dir "$COPLAYER_DATA_DIR" \
  --train.total-timesteps 50000 \
  --train.checkpoint-interval 1 \
  --train.render False \
  --train.minibatch-size 2912 \
  --train.max-minibatch-size 2912 \
  --train.minibatch-multiplier 1 \
  --train.batch-size 2912 \
  --train.horizon 91 \
  --env.num-maps 1 \
  --env.num-agents 32 \
  --env.scenario-length 91 \
  --env.conditioning.type all \
  --env.map-dir "$FIXTURE_MAPS" \
  --eval.human-replay-eval False \
  --eval.eval-interval 100000 \
  --vec.num-envs 1 --vec.num-workers 1 --vec.batch-size 1 \
  --tag smoke_coplayer >"$COPLAYER_LOG" 2>&1

# Locate the run we just made — there's only one inside the smoke data dir.
COPLAYER_DIR=$(ls -d "$COPLAYER_DATA_DIR"/puffer_drive_*/ 2>/dev/null | head -1 | sed 's:/$::')
test -n "$COPLAYER_DIR" || { echo "FAIL: no co-player run dir found"; tail -40 "$COPLAYER_LOG"; exit 1; }
COPLAYER_PT=$(ls -t "$COPLAYER_DIR"/model_*.pt | head -1)
echo "    -> $COPLAYER_PT"
test -f "$COPLAYER_DIR/info.json" || { echo "FAIL: info.json missing for co-player"; exit 1; }
echo "    -> info.json sidecar OK"

echo
echo "==> 3/5: training tiny adaptive ego against that co-player (k=2)"
ADAPTIVE_LOG="$SMOKE_DIR/adaptive.log"
puffer train puffer_adaptive_drive \
  --policy-architecture Recurrent \
  --rnn-name Recurrent \
  --train.data-dir "$ADAPTIVE_DATA_DIR" \
  --train.total-timesteps 50000 \
  --train.checkpoint-interval 1 \
  --train.render False \
  --train.minibatch-size 2912 \
  --train.max-minibatch-size 2912 \
  --train.minibatch-multiplier 1 \
  --train.batch-size 2912 \
  --train.horizon 182 \
  --env.num-maps 2 \
  --env.num-agents 16 \
  --env.num-ego-agents 8 \
  --env.k-scenarios 2 \
  --env.scenario-length 91 \
  --env.conditioning.type none \
  --env.co-player-enabled True \
  --env.co-player-policy.policy-path "$COPLAYER_PT" \
  --env.co-player-policy.architecture Recurrent \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub 0.1 \
  --env.co-player-policy.conditioning.discount-weight-lb 0.8 \
  --env.co-player-policy.conditioning.discount-weight-ub 1.0 \
  --env.map-dir "$FIXTURE_MAPS" \
  --eval.human-replay-eval False \
  --eval.eval-interval 100000 \
  --vec.num-envs 1 --vec.num-workers 1 --vec.batch-size 1 \
  --tag smoke_adaptive >"$ADAPTIVE_LOG" 2>&1

ADAPTIVE_DIR=$(ls -d "$ADAPTIVE_DATA_DIR"/puffer_adaptive_drive_*/ 2>/dev/null | head -1 | sed 's:/$::')
test -n "$ADAPTIVE_DIR" || { echo "FAIL: no adaptive run dir found"; tail -40 "$ADAPTIVE_LOG"; exit 1; }
ADAPTIVE_PT=$(ls -t "$ADAPTIVE_DIR"/model_*.pt | head -1)
echo "    -> $ADAPTIVE_PT"
test -f "$ADAPTIVE_DIR/info.json" || { echo "FAIL: info.json missing for adaptive"; exit 1; }
echo "    -> info.json sidecar OK"

echo
echo "==> 4/5: rendering all four modes"
RENDER_FAILURES=0

run_render() {
  local label="$1"; shift
  echo "    [$label]"
  if xvfb-run -a -s "-screen 0 1280x720x24" python render.py "$@" --num-renders 1 --num-maps 1 --map-dir "$FIXTURE_MAPS" 2>&1 | tail -5; then
    echo "      ok"
  else
    echo "      FAILED"
    RENDER_FAILURES=$((RENDER_FAILURES + 1))
  fi
}

run_render "baseline (co-player ckpt rendered solo)" \
  --model-path "$COPLAYER_PT" --conditioning-type all
run_render "coplayer (co-player + frozen co-player)" \
  --model-path "$COPLAYER_PT" --co-player-path "$COPLAYER_PT" \
  --conditioning-type all --co-player-conditioning-type all \
  --co-player-entropy-weight-lb 0 --co-player-entropy-weight-ub 0.1 \
  --co-player-discount-weight-lb 0.8 --co-player-discount-weight-ub 1.0
run_render "adaptive vs co-player (k=2)" \
  --model-path "$ADAPTIVE_PT" --co-player-path "$COPLAYER_PT" \
  --co-player-conditioning-type all \
  --co-player-entropy-weight-lb 0 --co-player-entropy-weight-ub 0.1 \
  --co-player-discount-weight-lb 0.8 --co-player-discount-weight-ub 1.0
run_render "human-replay (adaptive ego, log partners)" \
  --model-path "$ADAPTIVE_PT" --human-replay

# Sanity: confirm at least one mp4 landed under each run dir.
for d in "$COPLAYER_DIR/renders" "$ADAPTIVE_DIR/renders"; do
  count=$(ls "$d"/*.mp4 2>/dev/null | wc -l)
  echo "    $d: $count mp4(s)"
  test "$count" -ge 1 || RENDER_FAILURES=$((RENDER_FAILURES + 1))
done

if [ "$RENDER_FAILURES" -gt 0 ]; then
  echo "FAIL: $RENDER_FAILURES render mode(s) failed"
  exit 1
fi

echo
echo "==> 4.5/5: in-training render + human-replay eval (the actual prod path)"
INTRAIN_DATA_DIR="$SMOKE_DIR/intrain"
INTRAIN_LOG="$SMOKE_DIR/intrain.log"
mkdir -p "$INTRAIN_DATA_DIR"

# Train at frequent checkpoint/render/eval intervals so the run is short
# but exercises every periodic hook. xvfb wraps the entire command
# because train.render=True and eval.human_replay_eval=True both invoke
# the C raylib renderer in-process.
xvfb-run -a -s "-screen 0 1280x720x24" puffer train puffer_drive \
  --policy-architecture Recurrent \
  --rnn-name Recurrent \
  --train.data-dir "$INTRAIN_DATA_DIR" \
  --train.total-timesteps 30000 \
  --train.checkpoint-interval 5 \
  --train.render True \
  --train.render-interval 5 \
  --train.minibatch-size 2912 \
  --train.max-minibatch-size 2912 \
  --train.minibatch-multiplier 1 \
  --train.batch-size 2912 \
  --train.horizon 91 \
  --env.num-maps 1 \
  --env.num-agents 32 \
  --env.scenario-length 91 \
  --env.conditioning.type all \
  --env.map-dir "$FIXTURE_MAPS" \
  --eval.human-replay-eval True \
  --eval.eval-interval 5 \
  --eval.human-replay-num-agents 8 \
  --eval.human-replay-num-maps 1 \
  --eval.human-replay-num-rollouts 1 \
  --eval.num-maps 1 \
  --vec.num-envs 1 --vec.num-workers 1 --vec.batch-size 1 \
  --tag smoke_intrain >"$INTRAIN_LOG" 2>&1

INTRAIN_DIR=$(ls -d "$INTRAIN_DATA_DIR"/puffer_drive_*/ 2>/dev/null | head -1 | sed 's:/$::')
test -n "$INTRAIN_DIR" || { echo "FAIL: in-training run dir missing"; tail -40 "$INTRAIN_LOG"; exit 1; }

# (1) Periodic training renders should land under <run_dir>/renders/ named epoch_*.
TRAIN_RENDERS=$(ls "$INTRAIN_DIR"/renders/epoch_*_baseline_*.mp4 2>/dev/null | wc -l)
echo "    training renders : $TRAIN_RENDERS"
test "$TRAIN_RENDERS" -ge 1 || { echo "FAIL: no training renders produced"; exit 1; }

# (2) Human-replay renders also land under <run_dir>/renders/ named with human_replay.
EVAL_RENDERS=$(ls "$INTRAIN_DIR"/renders/epoch_*_human_replay_*.mp4 2>/dev/null | wc -l)
echo "    human-replay renders: $EVAL_RENDERS"
test "$EVAL_RENDERS" -ge 1 || { echo "FAIL: no human-replay renders produced"; exit 1; }

# (3) Human-replay eval scores (reported via wandb log) — the subprocess prints
# them between HUMAN_REPLAY_METRICS_START/END markers in its stdout, which gets
# captured by the parent's stdout (intrain.log).
if grep -q HUMAN_REPLAY_METRICS_START "$INTRAIN_LOG"; then
  echo "    human-replay metrics: ok"
else
  echo "FAIL: HUMAN_REPLAY_METRICS markers missing from $INTRAIN_LOG"
  exit 1
fi

echo
echo "==> 5/5: pytest suite"
pytest \
  tests/test_drive_conditioning.py \
  tests/test_map_dir_flow.py \
  tests/test_render_pipeline.py \
  tests/test_drive_config.py \
  --no-header -q

echo
echo "==> smoke OK"
echo "    co-player log : $COPLAYER_LOG"
echo "    adaptive log  : $ADAPTIVE_LOG"
echo "    co-player ckpt: $COPLAYER_PT"
echo "    adaptive ckpt : $ADAPTIVE_PT"
