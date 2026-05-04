#!/bin/bash
# Driver for the (discount_lb, entropy_ub) co-player grid sweep.
# Loops batches of 5 in parallel; called from nuplan_transformer_local_201_de_grid.sh.  set -u

cd /workspace/ADA
source .venv/bin/activate

DRIVER_LOG=/tmp/coplayer_de_grid_driver.log
echo "[$(date '+%H:%M:%S')] === DRIVER START ===" | tee $DRIVER_LOG

# Grid (entropy varies fastest)
RUNS=(
  "0.4 0.001"
  "0.4 0.01"
  "0.4 0.05"
  "0.4 0.1"
  "0.4 0.2"
  "0.6 0.001"
  "0.6 0.01"
  "0.6 0.05"
  "0.6 0.1"
  "0.6 0.2"
  "0.8 0.001"
  "0.8 0.01"
  "0.8 0.05"
  "0.8 0.1"
  "0.8 0.2"
)

# Fixed knobs (match existing 5-partner training)
COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=5000
SEED=42
CONTEXT_LENGTH=201
SCENARIO_LENGTH=201
MINIBATCH_SIZE=32160

read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3 4}"
N_GPUS=${#GPU_ARR[@]}
N_RUNS=${#RUNS[@]}

echo "[$(date '+%H:%M:%S')] $N_RUNS jobs, $N_GPUS gpus (${GPU_ARR[*]})" | tee -a $DRIVER_LOG

run_one() {
  local d=$1 e=$2 gpu=$3
  local tag=coplayer_de_grid_d${d}_e${e}
  local outlog=/tmp/coplayer_${tag}.log
  echo "[$(date '+%H:%M:%S')] LAUNCH d=$d e=$e gpu=$gpu (tag=$tag)" | tee -a $DRIVER_LOG
  CUDA_VISIBLE_DEVICES=$gpu xvfb-run -a puffer train puffer_drive \
    --wandb --wandb-project ada_coplayer_sweep \
    --tag $tag \
    --env.map-dir resources/drive/binaries/nuplan_201 \
    --env.num-maps $NUPLAN_NUM_MAPS \
    --env.scenario-length $SCENARIO_LENGTH \
    --env.reward-lane-align $LANE_REWARD \
    --env.conditioning.type all \
    --env.conditioning.collision-weight-lb $COLLISION_LB \
    --env.conditioning.collision-weight-ub 0 \
    --env.conditioning.offroad-weight-lb $OFFROAD_LB \
    --env.conditioning.offroad-weight-ub 0 \
    --env.conditioning.entropy-weight-lb 0 \
    --env.conditioning.entropy-weight-ub $e \
    --env.conditioning.discount-weight-lb $d \
    --env.conditioning.discount-weight-ub $DISCOUNT_UB \
    --policy-architecture Transformer \
    --train.context-length $CONTEXT_LENGTH \
    --train.horizon $CONTEXT_LENGTH \
    --train.minibatch-size $MINIBATCH_SIZE \
    --train.max-minibatch-size $MINIBATCH_SIZE \
    --train.learning-rate 0.003 \
    --train.checkpoint-interval 50 \
    --train.seed $SEED \
    --eval.map-dir resources/drive/binaries/nuplan_201 > $outlog 2>&1
  echo "[$(date '+%H:%M:%S')] DONE d=$d e=$e gpu=$gpu" | tee -a $DRIVER_LOG
}

# Run in batches of N_GPUS, sequential between batches.
for ((idx=0; idx<N_RUNS; idx+=N_GPUS)); do
  batch=$((idx/N_GPUS+1))
  echo "[$(date '+%H:%M:%S')] === BATCH $batch START ===" | tee -a $DRIVER_LOG
  PIDS=()
  for ((j=0; j<N_GPUS && idx+j<N_RUNS; j++)); do
    read -r D E <<< "${RUNS[$((idx+j))]}"
    GPU=${GPU_ARR[$j]}
    run_one "$D" "$E" "$GPU" &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  echo "[$(date '+%H:%M:%S')] === BATCH $batch DONE ===" | tee -a $DRIVER_LOG
done

echo "[$(date '+%H:%M:%S')] === ALL DONE ===" | tee -a $DRIVER_LOG
