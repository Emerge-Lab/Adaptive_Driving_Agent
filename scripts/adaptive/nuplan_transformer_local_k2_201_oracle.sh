#!/bin/bash
set -e

# k=2 adaptive ego training, oracle ego + entropy curriculum.
#
# Diagnostic single run: same config as curr_e0.5 (wandb hprfn8dc),
# but with --env.ego-is-oracle True. The ego now sees the partner's
# per-env conditioning vector appended to the END of its obs (5 slots
# after road_obs).
#
# Hypothesis: if the cache was unused because the policy couldn't infer
# partner type from behavior, then handing the partner's conditioning
# directly should produce a meaningfully adaptive policy (ada_delta
# rises significantly above ~0). If even the oracle policy doesn't
# adapt, the bottleneck is downstream (action conditioning).
#
# See notes/oracle_partner_conditioning_investigation.md for full
# investigation, design, and analysis plan.
#
# Defaults to GPU 6 (clean idle GPU; the k_eff runs sit on 4-5).
#
# Override GPU:    GPUS="6" bash <script>
# Stop:            tmux kill-session -t adaptive_local_k2_201_oracle

LABEL="oracle_curr_e0.5"
COPLAYER=experiments/puffer_drive_6rauydj2.pt
PARTNER_ID=6rauydj2

COLLISION_LB=-2
OFFROAD_LB=-2
LANE_REWARD=0.01
DISCOUNT_LB=0.4
DISCOUNT_UB=1
ENTROPY_UB=0.50
NUPLAN_NUM_MAPS=4999
SEED=42

K_SCENARIOS=2
SCENARIO_LENGTH=201
HORIZON=$((K_SCENARIOS * SCENARIO_LENGTH))     # 402

NUM_WORKERS=16
NUM_ENVS=16
MINIBATCH_MULTIPLIER=100
MAX_MINIBATCH_SIZE=40200

GPUS=${GPUS:-"6"}
read -r -a GPU_ARR <<< "$GPUS"
GPU=${GPU_ARR[0]}

SESSION=adaptive_local_k2_201_oracle
TAG="ada_${LABEL//./}_vs_${PARTNER_ID}_lane${LANE_REWARD}_k2_201"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "exp0_${LABEL//./}"

CMD="cd /workspace/ADA && \
source .venv/bin/activate && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo 'Running ${LABEL}: ego_is_oracle=True curr=True eub=${ENTROPY_UB} on GPU $GPU' && \
xvfb-run -a puffer train puffer_adaptive_drive \
  --wandb --wandb-project adaptive_aligned \
  --tag $TAG \
  --policy-architecture Transformer \
  --rnn-name Transformer \
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
  --env.co-player-policy.conditioning.entropy-weight-ub $ENTROPY_UB \
  --env.co-player-policy.conditioning.discount-weight-lb $DISCOUNT_LB \
  --env.co-player-policy.conditioning.discount-weight-ub $DISCOUNT_UB \
  --env.external-co-player-actions True \
  --env.map-rand-per-scenario False \
  --env.entropy-curriculum-enabled True \
  --env.ego-is-oracle True \
  --eval.map-dir resources/drive/binaries/nuplan_201 \
  --eval.human-replay-eval True \
  --eval.human-replay-num-rollouts 50 \
  --eval.eval-interval 5"

tmux send-keys -t "$SESSION:exp0_${LABEL//./}" "$CMD" C-m

echo
echo "Launched oracle adaptive run in tmux session '$SESSION' on GPU: $GPU"
echo "  ${LABEL}: ego_is_oracle=True + entropy curriculum (mirrors curr_e0.5 + oracle slot)"
echo "Attach: tmux attach -t $SESSION"
echo "Stop:   tmux kill-session -t $SESSION"
