#!/bin/bash
set -e

# k=2 adaptive ego training — γ sweep across 4 values × 2 partners.
#
# Why this experiment exists: prior runs all used γ=0.98 in adaptive.ini
# [train], which over horizon=402 gives 0.98^200 ≈ 0.018 cross-boundary
# credit and 0.98^402 ≈ 0.0003 end-to-end. Effectively zero credit
# assignment from s_1 to s_0 → policy can't learn cross-scenario
# adaptation. This was the silent killer of every previous adaptive
# experiment (curriculum, k_eff, oracle); see
# notes/oracle_partner_conditioning_investigation.md.
#
# What we want from this sweep:
#   1) confirm γ is THE bottleneck: high γ should produce ada_delta > 0
#   2) identify the sweet spot γ for k=2/201
#   3) replicate across two partners with different entropy regimes
#
# Simplest possible setup: NO entropy curriculum, NO oracle, NO k_eff
# curriculum, NO map_rand. Just plain adaptive training against a single
# partner with per-episode conditioning sampling, varying γ only.
#
# Default GPUs: 0-5 (6 runs, leaves headroom on GPU 6-7 for analysis
# scripts or a follow-up γ=0.998 run if 0.995 and 0.999 diverge).
#
# Override GPUs:    GPUS="0 1 2 3 4 5" bash <script>
# Stop everything:  tmux kill-session -t adaptive_local_k2_201_gamma_sweep

# (label, gamma, partner_id, entropy_ub) — 3 well-spaced gammas × 2 partners
RUNS=(
  "e01_g099   0.99   2e029h15 0.10"
  "e01_g0995  0.995  2e029h15 0.10"
  "e01_g0999  0.999  2e029h15 0.10"
  "e05_g099   0.99   6rauydj2 0.50"
  "e05_g0995  0.995  6rauydj2 0.50"
  "e05_g0999  0.999  6rauydj2 0.50"
)

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_LB=0.4
DISCOUNT_UB=1
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 402

NUM_WORKERS=32
NUM_ENVS=32
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"0 1 2 3 4 5"}
read -r -a GPU_ARR <<< "$GPUS"
N_RUNS=${#GPU_ARR[@]}
if [ "$N_RUNS" -gt "${#RUNS[@]}" ]; then
  echo "ERROR: more GPUs ($N_RUNS) than runs (${#RUNS[@]})" >&2
  exit 1
fi

SESSION=adaptive_local_k2_201_gamma_sweep
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n exp0

for ((i=0; i<N_RUNS; i++)); do
  read -r LABEL GAMMA PARTNER_ID EUB <<< "${RUNS[$i]}"
  GPU=${GPU_ARR[$i]}
  WIN="exp${i}_${LABEL}"
  TAG="ada_gsweep_${LABEL}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"
  COPLAYER="experiments/puffer_drive_${PARTNER_ID}.pt"

  if [ "$i" -eq 0 ]; then
    tmux rename-window -t "$SESSION:exp0" "$WIN"
  else
    tmux new-window -t "$SESSION" -n "$WIN"
  fi

  CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: gamma=${GAMMA} partner=${PARTNER_ID} (e_ub=${EUB}) on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
  --train.gamma $GAMMA \
  --train.horizon $HORIZON \
  --train.minibatch-multiplier $MINIBATCH_MULTIPLIER \
  --train.max-minibatch-size $MAX_MINIBATCH_SIZE \
  --train.cpu-offload True \
  --train.checkpoint-interval 10 \
  --train.render-interval 30 \
  --train.seed $SEED \
  --vec.num-workers $NUM_WORKERS \
  --vec.num-envs $NUM_ENVS \
  --env.map-dir resources/drive/binaries/nuplan_201 \
  --env.num-maps $NUPLAN_NUM_MAPS \
  --env.scenario-length $SCENARIO_LENGTH \
  --env.k-scenarios $K_SCENARIOS \
  --env.conditioning.type none \
  --env.reward-lane-align $LANE_REWARD \
  --env.co-player-enabled 1 \
  --env.co-player-policy.policy-path $COPLAYER \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.transformer.horizon $SCENARIO_LENGTH \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.collision-weight-lb $COLLISION_LB \
  --env.co-player-policy.conditioning.collision-weight-ub 0 \
  --env.co-player-policy.conditioning.offroad-weight-lb $OFFROAD_LB \
  --env.co-player-policy.conditioning.offroad-weight-ub 0 \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub $EUB \
  --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
  --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.human-replay-num-rollouts 50 \
  --eval.eval-interval 10"

  tmux send-keys -t "$SESSION:$WIN" "$CMD" C-m
done

echo
echo "Launched $N_RUNS gamma-sweep runs in tmux session '$SESSION' on GPUs: ${GPU_ARR[*]}"
echo "  e=0.10 partner (2e029h15): γ ∈ {0.99, 0.995, 0.999}"
echo "  e=0.50 partner (6rauydj2): γ ∈ {0.99, 0.995, 0.999}"
echo "Attach: tmux attach -t $SESSION"
echo "Stop all: tmux kill-session -t $SESSION"
