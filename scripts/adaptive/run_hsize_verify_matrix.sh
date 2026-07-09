#!/usr/bin/env bash
# Driver for hidden_size × fastpath verification matrix.
# Runs sequentially inside an existing GPU alloc (via srun --jobid --overlap).
# For each row: launches run_puffer_hsize_test.sh, captures exit code + SPS hits.

set -u
LOG_ROOT="${LOG_ROOT:-logs/hsize_verify}"
TIMEOUT_SEC="${TIMEOUT_SEC:-240}"
SUMMARY="$LOG_ROOT/SUMMARY.tsv"
mkdir -p "$LOG_ROOT"

# Header for summary (overwrites)
printf "row\thidden_size\tfix\texit_code\tsaw_sps\tfirst_sps_line\tcrash_signal\n" > "$SUMMARY"

# Matrix: hidden_size FIX_FASTPATH
# Run 2: bug already confirmed bs-shape (h=64 plain crashed with MHA error).
# Now we only need to prove FIX actually lets training make progress (SPS > 0).
ROWS=(
  "64  1"   # fix at small → expect real SPS > 0
  "128 1"   # fix at 128 → expect real SPS > 0
  "256 1"   # fix at original size → CRITICAL: expect real SPS > 0
  "512 1"   # fix at large → expect real SPS > 0
)

ROW_IDX=0
for ROW in "${ROWS[@]}"; do
  ROW_IDX=$((ROW_IDX + 1))
  HSIZE=$(echo "$ROW" | awk '{print $1}')
  FIX=$(echo "$ROW" | awk '{print $2}')
  TAG="row${ROW_IDX}_h${HSIZE}_fix${FIX}"
  LOG="$LOG_ROOT/${TAG}.log"

  echo "==================================================================" | tee -a "$SUMMARY.console"
  echo "[matrix] row=$ROW_IDX  hidden=$HSIZE  fix=$FIX  timeout=${TIMEOUT_SEC}s  log=$LOG" | tee -a "$SUMMARY.console"
  echo "==================================================================" | tee -a "$SUMMARY.console"

  bash scripts/adaptive/run_puffer_hsize_test.sh "$HSIZE" "$FIX" "$LOG" "$TIMEOUT_SEC"
  EC=$?

  # Detect a REAL non-zero SPS. Puffer renders "SPS  0" in initial dashboard,
  # which doesn't prove progress. Look for SPS > 0.
  FIRST_SPS=$(grep -E 'SPS[[:space:]]+[0-9]' "$LOG" 2>/dev/null | grep -vE 'SPS[[:space:]]+0[[:space:]]' | head -1 | tr -d '\t\n' | cut -c1-200)
  SAW_SPS=0
  [[ -n "$FIRST_SPS" ]] && SAW_SPS=1

  # Detect crash markers
  CRASH=""
  if grep -qE 'CUDA error|illegal memory access|RuntimeError|Segmentation fault|core dumped' "$LOG" 2>/dev/null; then
    CRASH=$(grep -E 'CUDA error|illegal memory access|RuntimeError|Segmentation fault|core dumped' "$LOG" | head -1 | tr -d '\t\n' | cut -c1-200)
  fi

  printf "%d\t%s\t%s\t%d\t%d\t%s\t%s\n" "$ROW_IDX" "$HSIZE" "$FIX" "$EC" "$SAW_SPS" "$FIRST_SPS" "$CRASH" >> "$SUMMARY"
  echo "[matrix] row=$ROW_IDX done  exit=$EC  saw_sps=$SAW_SPS  crash=${CRASH:0:80}" | tee -a "$SUMMARY.console"
done

echo "" | tee -a "$SUMMARY.console"
echo "===== FINAL SUMMARY =====" | tee -a "$SUMMARY.console"
column -t -s $'\t' "$SUMMARY" | tee -a "$SUMMARY.console"
