# Paper analysis — living document

Maintained by Claude; updated whenever new results land. Started 2026-07-03.
Companion: `notes/paper_plan.md`, memory `project_paper_goal_significant_adaptation.md`.

## Identity of the paper

AdA-style human-timescale adaptation (arXiv 2301.07608) demonstrated in a **real-world,
safety-critical domain with real human data** (nuplan driving logs), at ~1/1000th AdA's
scale, with a **mechanism decomposition** AdA doesn't have (adaptation ≈ crash reduction),
and a **demonstration-prompting extension** AdA explicitly leaves open (they prompt an
agent never trained with demos; we also train with demos).

## Headline results (as of 2026-07-03)

1. **60-cell grid** (4 entropy_ub × 5 k × 3 seeds, 540 maps × 20 rollouts, per-trial return):
   k=4 sweet spot at low/mid entropy. Best cells on adaptable maps (p₀<0.8):
   0.20/k4 ΔR=+1.36 (n=62), 0.10/k4 +0.89 (n=106, cleanest cross-seed band).
   k=2 actively degrades; k=5 anomalously flat (suspected undertraining).
2. **Decomposition** (env-side per-trial accumulators): ~80–90 % of ΔR is crash reduction,
   ~10–15 % goal-reach, lane ≈ 0. "Agent stops repeating its crashes."
3. **Hidden-size ablation** (k=4/e=0.10, 3 seeds each): eval return 0.293 (h64) → 0.421 (h128)
   → 0.453 (h256). h=512 OOM'd at training (37.7 GiB alloc). Capacity needed, saturating.
4. **Human-demo training** (h=128 × 3 seeds, trial 0 = human replay, in flight, job 12226479):
   training metrics t0=0.93 (human, fixed), t1≈0.90, t2/t3≈0.89 — all seeds agree to ~3 decimals.
   Post-demo first attempt nearly matches the demo. Mirrors AdA §3.8 result shape
   (prompted < teacher, prompted > unprompted — second comparison pending offline eval).

## AdA parity map

| AdA element | Ours | Status |
|---|---|---|
| k-trial episodes, memory across trials, task reset at boundary | identical | ✓ |
| Score = per-trial reward normalized per task | raw return; **should human-normalize** | TODO |
| Zero-shot = trial-1, few-shot = improvement w/ trials | same framing available | ✓ |
| 1000 held-out procedural + 30 probe tasks | 539 eval maps but **NOT held out** (see issue 1) | ⚠ |
| Median + 20th-pct aggregation | mean ± std currently | TODO |
| Memory-architecture ablation | none (fixed 2-layer TF) | won't do |
| Curriculum ablation (no-op filter, PLR) | none yet — **in progress** (see curriculum plan) | TODO |
| Scale ablation 6M→265M | h 64/128/256 (~0.2M–1.2M) | partial |
| Memory-length ablation | horizon already spans episode | n/a — note in text |
| Distillation | none | won't do |
| Human comparison (30 probe tasks, median score vs trials) | human log return per map = free baseline | TODO (cheap) |
| §3.8 demo prompting (teacher plays trial 1) | demo-at-eval arm (A) = exact replica; demo-trained arm (B) = novel extension | in flight |

## Ranked reviewer risks

1. **[SETTLED 2026-07-03] Eval maps are not held out — accepted by design.**
   `nuplan_hard` is a symlink subset of `nuplan_201` (top-10 % by SDC-interaction
   density). USER DECISION: nuplan_hard STAYS the primary eval — hardness
   (intersections / interaction pressure) is the point; a random held-out split
   would be dominated by straight-road maps with nothing to adapt to.
   Handling: (a) phrase claims as in-context ADAPTATION (valid regardless of
   training exposure), never "held-out generalization"; (b) optional appendix
   confirmation on the hard subset of the 403 never-trained maps (ids ≥ 4999,
   `nuplan_heldout_403` built, ~40 hard maps by interaction threshold) — hard AND
   unseen; (c) curriculum continuation excludes nuplan_hard originals from the
   frontier training dir for comparison integrity (frontier filter would otherwise
   concentrate training on the eval set).
2. **[CRITICAL] Adaptable-map filter may have selection bias.** If the p₀<0.8 selector
   uses the same rollouts as the ΔR = t_last − t₀ measurement, regression-to-the-mean
   inflates ΔR. Memory claims "independent selector" (binary-success eval vs return
   re-eval — different SLURM runs, so plausibly independent rollouts) — **verify**, and
   if not independent do split-half (select on rollouts 1–10, measure on 11–20).
3. **[CRITICAL] No memory-ablation causal control.** Run headline-cell evals with
   `RECOVERY_CACHE_RESET_PER_SCENARIO=1` (evaluator.py ~L701). Flat curve under reset =
   causal proof that cross-trial memory drives the improvement. Cheap, decisive.
4. **Normalization + aggregation.** Adopt human-normalized score (return ÷ human-log
   return per map) and AdA-style median/20th-pct plots. Human log return obtainable
   from a demo-mode eval (trial 0 = human replay ⇒ t₀ = human performance per map).
5. **Anomalies.** k=2 degradation (opposite of explore-exploit signature — discuss);
   k=5 flat (resolve undertraining or drop k=5 with justification).
6. **No curriculum** — AdA core ingredient. In progress; see curriculum plan below.
7. **Budget asymmetries.** h256/s42 iter=80; demo seeds 43/44 short of 3B (walltime).
   Rerun or footnote consistently.
8. **Cross-seed per-map irreproducibility** (r≈0–0.25) — frame all claims at
   population level; never per-map. Already understood; keep out of claims.
9. **In-training eval bug** (hidden-size not propagated to eval subprocess — fixed
   2026-07-03 in utils.py): all non-h256 runs have no in-training eval curves on wandb;
   offline evals are source of truth.

## Curriculum plan (started 2026-07-03)

Motivation: uniform map sampling → >54 % of nuplan_201 maps have zero ego-other
interaction (`notes/nuplan_hard.md`); saturated maps dilute adaptation pressure.
AdA: no-op filter and PLR both strongly beat uniform; PLR wins at high trial counts.

**Phase 1 — static frontier filter (running first):**
- Score all training maps with a trained baseline checkpoint (per-map zero-shot
  goal-reach t₀, few rollouts each).
- Keep maps with t₀ in a band (~0.1–0.8: not impossible, not saturated); build
  renumbered symlink dir `resources/drive/binaries/nuplan_201_frontier/`
  (same mechanism as nuplan_hard; naming must stay `map_%03d.bin` contiguous).
- Retrain h=128 × 3 seeds with only `--env.map-dir` changed → matched A/B against
  existing h=128 ablation cells. Eval on the standard 540-map suite.
- Consider 80/20 frontier/uniform mixture if pure-frontier destabilizes basic driving.

**Phase 2 — online PLR-lite (if Phase 1 moves the needle):**
- Worker-local priority sampling at map re-init (`_reinit_envs_with_new_maps` in
  drive.py currently samples uniformly): per-worker EMA of per-map success, sample
  ∝ (1 − success_EMA). No cross-worker sync (32 workers build independent tables —
  accepted approximation). ~40–60 lines in drive.py only.
- This is the AdA-parallel ablation (uniform vs static filter vs online prioritized).

## Figure plan

- Fig 1: per-trial human-normalized score curves (entropy×k grid or best-cells row),
  median + 20th pct, adaptable maps + all maps inset.
- Fig 2: ΔR decomposition (crash/goal/lane bar per cell — have `g_dR_breakdown_*`).
- Fig 3: modulators — k sweet spot, entropy, capacity (hsize), curriculum (pending).
- Fig 4: demonstration prompting 3-way (no-demo / demo-at-eval / demo-trained) vs
  human line — the AdA-§3.8-extension money figure.
- Fig 5 (control): memory-reset counterfactual — flat curve.

## Pending experiments / jobs

- 12226479 demo training: cells 0 (done 3.8B), 1 (walltime ~2.4B), 2 running (1.6B @ 11h).
  Watcher `b1dft1qeo` fires when queue drains → offline 3-way eval
  (`cluster_eval540_demo.sh`; arm-B wids: gmfve041 s42, gxm7jb7t s43, cell-2 wid TBD).
- Then: selector-independence check; cache-reset control; held-out map split check;
  curriculum Phase 1.

## Result log (newest first)

- 2026-07-09: **MEMORY-ABLATION CONTROL LANDED — MECHANISM PROVEN CAUSALLY**
  (array 13142075, fixed per-agent trial-boundary reset; fig
  `outputs/eval540_cachereset2/g_memory_ablation.jpg` = Fig 5).
  Adaptable maps: 0.10/k4 intact ΔR +0.93 vs reset **+0.08**; 0.20/k4 intact
  +1.32 vs reset **−0.08**. Reset curves flat (trials iid), trial-0 levels match
  intact within noise (correct sanity). Reviewer risk #3 CLOSED. Combined with
  e0001 anchor + demo results: adaptation = cross-trial transformer memory of
  the agent's own failures — causal, triangulated from three directions.
  Implementation: models.py per-agent transformer_position + evaluator.py
  trial-mode reset (unit tests in scripts/adaptive/verify_per_agent_reset.py).
- 2026-07-08: **CURRICULUM RESULT = NULL-NEGATIVE** (eval 12952198, figs
  g_curriculum_doseresponse.jpg). ΔR adaptable maps (n=106, 0.10-selector):
  parent +0.93±0.23 > uniform+1B +0.73±0.51 > interaction+1B +0.63±0.41 >
  frontier+1B +0.58±0.23. Zero-shot all-maps also drifts down (0.23 → 0.16/0.20/0.11).
  Reads: (1) +1B continuation at low resumed LR does NOT amplify adaptation;
  (2) part of the drop is generic more-training-erodes-ΔR (uniform control also
  drops — matches the old entropy-curriculum motivation observation);
  (3) frontier-specific extra drop is within seed noise; ordering is OPPOSITE the
  dose-response prediction. Also: frontier training did NOT transfer to the
  (excluded) eval maps — t0 on adaptable eval maps unchanged (−1.76→−1.79).
  Paper framing: honest negative — "brief fine-tuning on failure sets neither
  amplifies in-context adaptation nor transfers zero-shot"; from-scratch
  curriculum (AdA-style PLR during full training) remains future work.
- 2026-07-08: **ENTROPY ANCHOR LANDED — MECHANISM ANSWERED** (eval540_e0001,
  fig g_entropy_tradeoff_5pt.jpg). Deterministic partner (e_ub=0.001): zero-shot
  all-maps +0.25, adaptation ΔR +0.43 (curve −1.34/−1.34/−1.14/−0.91, 0.10-selector).
  **Adaptation SURVIVES with nothing to infer about the partner** → own-failure
  memory is the base mechanism (~half the peak effect); partner stochasticity
  amplifies it (0.43 → 0.89/1.36 at 0.10/0.20) before destroying it (0.50: 0.35).
  Caveats: anchor uses miku2puk partner + 0.10-selector keep set; own-selector
  number pending if needed.
- 2026-07-05: **CACHE-RESET CONTROL WAS A NO-OP — result invalid, do not cite.**
  Array 12538732 (6 cells) produced curves identical to memory-intact (ΔR +0.97/+1.27
  vs +0.93/+1.32) BUT the reset code (evaluator.py:785) only exists in the
  SCENARIO-mode branch; trial mode (gb=3, our evals) never resets. Banner also
  invisible (eval_final_540 swallows child stdout). Mechanism claim remains
  UNTESTED. TODO: implement per-agent reset at trial boundaries in the trial-mode
  loop (async boundaries; state is shared batch tensors; must check legacy
  forward_eval semantics for zeroed history rows / garbage mask), verify on one
  cell, rerun the control. Outputs/eval540_cachereset currently = duplicate plain
  evals (useful only as a rollout-noise replicate: cell-level ΔR reproduces within
  ±0.05 across independent 20-rollout evals — actually cite-worthy for noise-floor).
- 2026-07-05: Curriculum arms status — COMPLETE: hard s42 (7m6lu6z1), hard s43
  (s4h7at8q), uniform s42/s43 (wids in logs 12414868_3/_4), frontier s43
  (12464239_1). INCOMPLETE (bad nodes gh005/gh014 — crawled at 2.5K SPS or
  cancelled): hard s44, uniform s44 (nothing saved), frontier s42 (saved
  epoch120/3.16B as h29ja02n). Finisher array **12530314** submitted (24h wall;
  site policy rejects --exclude, monitoring SPS manually). frontier s44 running;
  e0001 anchors s42/s43 running, s44 queued.
- 2026-07-04: ENTROPY-ANCHOR LAUNCHED (**12465000**, 3 seeds): e_ub=0.001 with
  partner miku2puk (deterministic extreme of its trained range), from-scratch k=4
  h=256 3B, tag e0001_anchor_k4. Purpose: left anchor of trade-off figure + MECHANISM
  TEST (deterministic partner: if ΔR preserved → own-failure memory drives adaptation;
  if flat → partner-inference). Interleaves with curriculum arms (user accepted delay).
  Queue at launch: hard s43/s44 + uniform s42 running; uniform s43/s44, frontier ×3,
  e0001 ×3 pending.
- 2026-07-04: **ENTROPY TRADE-OFF FIGURE** (`outputs/eval540_return/g_entropy_tradeoff.jpg`,
  candidate Fig 3 panel): at k=4, zero-shot return rises monotonically with partner
  entropy (0.34/0.24/0.41/0.53 for 0.05/0.10/0.20/0.50 — domain-randomization
  robustness) while adaptation ΔR on adaptable maps peaks at moderate entropy
  (+0.71/+0.89/+1.36/+0.35) — partner must be stochastic enough to matter,
  predictable enough to infer. AdA "task richness" analog with a cleaner dial.
  Also from grid review: k=5/k=6 ran the memsafe recipe (different batch geometry;
  lower absolute t0 returns) → k=5-flat anomaly is CONFOUNDED; scope k-claims to
  {2,3,4,6} or rerun k5. Five 0.10-column cells stopped at 61–75 % of target iters.
- 2026-07-04: FRONTIER ARM LAUNCHED (**12464239**, 3 seeds): nuplan_201_frontier =
  940 maps with mean t0 < 0.9 (from scoring sweep; eval + heldout excluded).
  Completes 3-arm curriculum ladder uniform(21% frontier density) /
  interaction(24%) / frontier(100%) — dose-response design, AdA
  uniform→no-op→PLR parallel. USER DECISIONS: no trial-0 PPO mask fix (no demo
  retraining planned); demo thread closes with arm-A result.
- 2026-07-04 (adaptable-map view, n=106, figs g_demo_2way_room / _score_room):
  baseline return −0.93→−0.49 vs demo-arm ★0.89 | −0.76→−0.56; success baseline
  0.762→0.778 vs demo ★0.943 | 0.734→0.758. Human-agent gap ≈ 1.8 return units on
  adaptable maps. A human demo buys ≤ one self-trial (return) and nothing in
  success; arms converge by trial 3. PAPER CLAIM: adaptation is driven by memory
  of the agent's OWN failures — an expert's successful trajectory carries no
  information about where the agent would crash (consistent with crash-reduction
  mechanism + demo-training collapse).
- 2026-07-04: **3-WAY DEMO EVAL LANDED (12463439) — headline surprise.** Mean per-trial
  return, 539 nuplan_hard maps, 3 seeds (fig `outputs/eval540_demo/g_demo_3way.jpg`):
  | arm | t0 | t1 | t2 | t3 |
  | C baseline no-demo | 0.396 | 0.380 | 0.435 | 0.473 |
  | A baseline+demo@eval | 0.864* | 0.420 | 0.444 | 0.454 |
  | B demo-trained+demo@eval | 0.864* | **−0.113** | 0.041 | 0.170 |
  (*t0 in demo arms = the human replay itself = 0.864 → this is also the
  human-normalization denominator, free.)
  Reads: (1) zero-shot prompting (arm A, AdA §3.8 protocol) is NEUTRAL — small t1
  bump vs C.t0 (+0.02, within seed noise), no lasting gain; differs from AdA who saw
  gains (their demos were fine-tuned-teacher = on-distribution; ours are human logs +
  hard-tail eval). (2) Demo-TRAINING is HARMFUL under distribution shift, and
  monotonically worse with more training (s42/3.8B t1=−0.37; s44/3.0B +0.07;
  s43/2.4B −0.04; 14–23 % of maps crash-dominated at t1) — consistent with
  imitation-collapse: training distribution is 75–80 % saturated maps where copying
  the demo works (training t1≈0.90 mirage), copying fails on interaction-dense maps.
  (3) Demo-trained arm has the steepest within-episode recovery slope (−0.11→0.17)
  — it adapts, from a crater.
  **Suspected training artifact to fix before rerunning:** during demo trial 0 the
  policy's sampled actions go into the PPO buffer but the env executes move_expert —
  gradient flows through actions that never drove the dynamics (off-policy
  contamination on trial-0 tokens). FIX: mask trial-0 steps out of the policy loss
  when demo_trial_0 is on. Alternative/complementary: train demos on HARD maps
  (link to curriculum thread).
- 2026-07-04: Demo training 12226479 DONE: gmfve041 s42 3.8B/iter114,
  gxm7jb7t s43 2.4B/iter90 (walltime), kualhrgw s44 3.0B/iter114. Final training
  trial scores ≈ t0 0.93 / t1 0.90 / t3 0.89 across seeds. 3-way offline eval
  launched (array **12463439**: arm A baseline+demo-at-eval, arm B demo-trained;
  arm C baseline-no-demo already in outputs/eval540_hsize). Curriculum arms:
  12414868_0 (hard s42) ~12h in, _1 started, _2-5 queued.
- 2026-07-03: Curriculum continuation LAUNCHED (array **12414868**): interaction-based
  filter chosen over checkpoint-performance filter (policy-independent, zero extra
  compute — reused scripts/nuplan_201_hardness_scores.csv). Built
  `nuplan_201_hardtrain` = 753 maps (sdc_interaction_steps ≥ 21 ≈ top 25 %, minus 540
  eval originals, minus ids ≥ 4999). 6 cells: hard vs uniform-control × 3 seeds,
  resume k4/e010 h256 parents (qxw6c0jh/ufmegw4l/jsckmpha) +1B steps, NEW wandb runs
  (branching two arms from one parent forbids --load-id: run-history + exp-dir
  collisions). Checkpoint-scoring array 12413720 running in parallel — will
  cross-check saturation of interaction-hard maps when it lands.
- 2026-07-03: CONFIRMED ids ≥ 4999 never trained (binding.h:280 `rand() % num_maps`,
  num_maps=4999) → built `nuplan_heldout_403` (403 truly unseen maps) + 10 scoring
  chunk dirs. Map-scoring array **12413720** launched (3 k=4 wids × 10 chunks,
  5 rollouts) → outputs/map_scoring/. Curriculum switched to CONTINUE-training design:
  resume k=4/e=0.10 h=256 checkpoints +1B steps, frontier vs uniform-control arms,
  frontier dir will exclude nuplan_hard originals + heldout ids.
- 2026-07-03: nuplan_hard ⊂ nuplan_201 discovered (issue 1). Curriculum plan drafted.
- 2026-07-03: demo training t1≈0.90 vs t0=0.93 stable across 3 seeds.
- 2026-07-02: eval-subprocess hidden-size bug found+fixed (utils.py); demo_trial_0
  implemented in drive.h/binding.c/drive.py + adaptive.ini; demo training launched.
- 2026-07-01: hsize eval: 0.293/0.421/0.453 (h64/128/256); h512 no data (OOM).
- 2026-06-30: hsize ablation trained (12004810); MHA fastpath fix moved into pufferl.py.
- 2026-06-25: ΔR decomposition → crash reduction is the mechanism.
- 2026-06-24: return metric locked; 60-cell grid headline (k=4 sweet spot).
