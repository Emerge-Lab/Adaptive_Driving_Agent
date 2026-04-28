# Paper TODO

## Pending — resume after recovery eval finishes
- **Resume the 4 k=3 adaptive runs** (paused on 2026-04-28 to free GPUs 4-7
  for parallel recovery-metric eval). Latest checkpoints are at epochs
  150-170. One-command resume:
  ```
  bash /workspace/ADA/scripts/adaptive/nuplan_transformer_local_k3_resume.sh
  ```
  The script already auto-locates the latest `model_*.pt` per wandb id and
  uses the optimized flags (cpu_offload + external_co_player_actions, nw=32).

## In progress
- **Conditional recovery metric** in `HumanReplayEvaluator`:
  per-(agent, scenario) success tracking, then `P(success in s_k | failed s_0)`.
  Surfaces in-context adaptation in the small fraction of nuplan scenes
  where adaptation actually matters (most scenes are easy and ego trivially
  succeeds in s_0, washing out the averaged `ada_delta_score`).

## Backlog (after we have a baseline conditional-recovery number)

- **Map rotation per scenario within an episode**.
  At each scenario boundary, swap the underlying nuplan map so the ego sees
  a new scene with new humans. KV cache preserved across scenarios so past
  observations can inform the new scene. Forces more "hard" cases by
  guaranteeing each scenario is genuinely novel. Implement as a flag:
  `--env.map-rand-per-scenario {none, eval, train, both}`. Likely requires
  retraining adaptive agents to handle the within-episode discontinuity.

- **Train-time conditional-recovery loss/weighting**: bias rollout sampling
  toward "hard" scenarios (rare but informative) once we have per-agent
  difficulty estimates from offline eval.

- **Cross-co-player generalization eval**: train ego against partners
  {A, B, C}, eval against held-out partner D. Tests whether learned
  adaptation generalizes to unseen partner styles.

- **Architecture ablation**: run with KV-cache disabled (or context-length=1)
  to confirm the cache is what's enabling adaptation (when it works).
