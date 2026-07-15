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
| Score = per-trial reward normalized per task | human-normalized (return ÷ human-log return; `outputs/eval540_norm/`) | ✓ 2026-07-10 |
| Zero-shot = trial-1, few-shot = improvement w/ trials | same framing available | ✓ |
| 1000 held-out procedural + 30 probe tasks | 539 eval maps but **NOT held out** (see issue 1) | ⚠ |
| Median + 20th-pct aggregation | done for headline cells (`g_human_norm_curves.jpg`) | ✓ 2026-07-10 |
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
2. **[SETTLED 2026-07-10] Adaptable-map selector bias is negligible.** Split-half
   analysis on fresh 20-rollout evals with raw per-(map,rollout) dumps
   (`outputs/eval540_splithalf/splithalf_summary.csv`): selecting on rollouts 0–9
   and measuring ΔR on 10–19 (and the reverse) gives ΔR +0.94/+0.89 (0.10/k4) and
   +1.24/+1.30 (0.20/k4) vs full-sample +1.00/+1.37 — bias ≤ ~0.1, ≈10 % of the
   effect. Cite the disjoint-half numbers in the robustness appendix. Fresh evals
   also independently replicate the original headline ΔR (rollout-noise replicate #2).
3. **[SETTLED 2026-07-10] Memory-ablation causal control LANDED — adaptation is
   causally memory-driven.** Fixed per-agent trial-mode reset (array 13142075,
   `outputs/eval540_cachereset2/`): reset curves FLAT (ΔR +0.07 / −0.08 on
   0.10/k4 and 0.20/k4 adaptable maps) vs intact (+0.92 / +1.32), with matching
   trial-0 anchors. Verified independently from raw CSVs on the vast box +
   `verify_per_agent_reset.py` unit tests pass. Fig 5 done (`g_memory_ablation.jpg`).
4. **Normalization + aggregation.** Adopt human-normalized score (return ÷ human-log
   return per map) and AdA-style median/20th-pct plots. Human log return obtainable
   from a demo-mode eval (trial 0 = human replay ⇒ t₀ = human performance per map).
5. **Anomalies.** k=2 degradation (opposite of explore-exploit signature — discuss);
   **k=5 flat: RESOLVED 2026-07-12 — pure memsafe-recipe artifact.** Standard-recipe
   rerun (3 seeds, 3B) gives adaptable-map ΔR +0.86 (all seeds positive) vs the
   memsafe cells' +0.09 — k=5 adapts on par with k=4 (+0.92). Scope k-claims to the
   standard-recipe cells; the sweet-spot narrative softens to "robust for k≥3".
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

**Phase 2 — online PLR-lite: IMPLEMENTED 2026-07-10 (from-scratch redesign
after the fine-tuning null-negative).** Worker-local priority sampling, env-side:
- C: `binding.shared(map_sample_weights=[...])` — optional per-map weights,
  silent uniform fallback (unit-tested: concentration, fallbacks, skew matches
  theory 0.92/0.92).
- Python (drive.py): per-map EMAs updated at episode boundaries from
  trial_ended/trial_R_goal; two scorers behind `--env.map-curriculum-mode`:
  `success` = ∝(1−trial-0-success EMA) (vanilla PLR-ish) and `gap` = ∝EMA of
  within-episode success gap t_last−t0 (**adaptation-frontier prioritization —
  novel vs AdA**). `uniform_floor` mixes uniform mass (no starvation),
  `optimism` scores unvisited maps (exploration). Logs under map_curriculum/*.
  No cross-worker sync (accepted approximation).
- Plan: 2 arms × 1 seed from scratch (success, gap) on the k5 GPUs when they
  free (~2026-07-12), uniform control = existing headline cells (free);
  scale winner to 3 seeds. This is the AdA-parallel uniform→filter→PLR ladder
  with from-scratch training (the July fine-tuning attempt was the weakest
  dose — see 07-08 null result).
- **Related-work positioning for the gap scorer (verified against AdA App.
  D.5, 2026-07-13):** AdA's PLR = modified Robust PLR with regret-approximating
  fitness via TD-error (ablated vs value-model / dynamics-model error); their
  no-op filter compares vs a do-nothing policy. NOTHING in AdA uses
  within-episode trial improvement. Our gap scorer = an in-context-regret
  estimate: the agent's own memory-adapted self is the regret "antagonist"
  (trial-k performance ≈ achievable, trial-0 = current). Frame as an
  instantiation of regret-based UED fitness (PAIRED 2012.02096, Robust PLR
  2110.02439, ACCEL 2203.01302) crossed with learning-progress curricula
  (TSCL 1707.00183, ALP-GMM 1910.07224, Oudeyer LP) transposed from
  parametric to in-context progress — NOT as a new paradigm. Motivation
  quote from AdA §3.3: regret-based PLR "especially helpful for learning
  longer-term adaptation".

## Figure plan — ALL BUILT 2026-07-13 (`scripts/adaptive/make_paper_figures.py`
→ `outputs/paper_figs/fig{1..5}_*.{pdf,jpg}`; rerun any subset by name)

- Fig 1 ✓ human-normalized per-trial score, both headline cells, median +
  20th pct, adaptable + all maps, human=1 line.
- Fig 2 ✓ ΔR decomposition. IMPORTANT measurement note discovered while
  building: env-side component accumulators are only snapshotted at trial-end
  flags, so maps whose final trial is horizon-truncated (~13–18/539 per seed,
  crash-heavy) have undefined t3 components — per_map_R and component-sum
  disagree ONLY on those maps/trial. Fig 2 therefore uses adaptable maps with
  complete final trials (n=76/41), where components reproduce R exactly;
  totals there are +1.05/+1.66 (higher than all-adaptable +0.92/+1.32 since
  truncated maps drag t3). Crash-avoidance share ≈ 70–77 % of ΔR.
- Fig 3 ✓ modulators: (a) k sweep at e_ub=0.10, single standard-minibatch
  series k2→k5 (k5 = rerun; USER DECISION 2026-07-13: memsafe k5/k6 cells
  removed from the figure entirely — superseded/confounded; caption notes
  the k5 recipe). Recipe taxonomy for the text: standard = vec32 +
  minibatch 50×H; memsafe (old k5/k6) = vec8 + HALVED minibatch 25×H
  (same LR) — flat at k5 (+0.09) but partially adapting at k6 (≈+0.67,
  noisy), so frame as "superseded", not "half minibatch kills adaptation";
  rerun = vec8 + full 50×H via grad accumulation. (b) entropy trade-off,
  (c) capacity. PLR panel to be added when arms finish (~07-14).
- Fig 4 ✓ demo 3-way vs human star/line, all-maps + adaptable panels.
- Fig 5 ✓ memory-reset control, restyled consistently from raw CSVs.

## Pending experiments / jobs

- 12226479 demo training: cells 0 (done 3.8B), 1 (walltime ~2.4B), 2 running (1.6B @ 11h).
  Watcher `b1dft1qeo` fires when queue drains → offline 3-way eval
  (`cluster_eval540_demo.sh`; arm-B wids: gmfve041 s42, gxm7jb7t s43, cell-2 wid TBD).
- Then: selector-independence check; cache-reset control; held-out map split check;
  curriculum Phase 1.

## Result log (newest first)

- 2026-07-14: 0.20-cell seed 45 (b7itnfmg, vec8) done + evaled: own-selector
  ΔR **+0.89** (n=62, −3.68→−2.79), zs all-540 +0.11 — the 0.20 cell keeps a
  strong adaptation signal at vec8 (unlike the 0.10-cell vec8 pair's
  {+0.56,+0.12}). Fixed-vec32-selector view −0.23 (selector mismatch again —
  own-selector is the valid view). vec32 trio ref +1.32. Also: PLR-success
  s44 launched (gia12pd5, GPU3) — both arms now 3 seeds in flight/done.
- 2026-07-14: **PLR 3-WAY FIRST RESULT (1 seed/arm) — both arms ≥ uniform,
  distinct signatures.** Own-selector, vec8-matched control
  (outputs/eval540_plr/): PLR-success zs +0.15 / ΔR +0.68 / succ 0.46→0.43
  (declining!); PLR-gap zs +0.02 / ΔR +0.58 / succ 0.34→0.47 (best success
  climb); uniform vec8 pair zs ≈0 / ΔR {+0.56,+0.12} / succ +0.13/+0.06.
  Signatures: success-scorer buys ZERO-SHOT (difficulty oversampling →
  stronger base policy, smallest adaptable set n=56); gap-scorer buys
  ADAPTATION (biggest within-episode success climb). Curriculum mechanics
  verified on wandb: gap concentrates sampling (weight entropy 8.51→7.55)
  ~2.5× more than success (→8.28); both visit ~all 4999 maps. Seed-43 arms
  launched for both (GPUs 0/1) to firm this up; k6 vec8 smoke on GPU2.
- 2026-07-14: Task C seed 46 (ey3ia9c5, 0.10/k4 vec8) done + evaled. Own-
  selector ΔR **+0.12** (weak); s45 +0.56 → the vec8 k4 pair is
  {+0.56, +0.12}, well below the vec32 trio {+1.26, +0.71, +0.78}. Combined
  with k5-vec8 adapting strongly (+0.86 mean), the picture is: small rollout
  batch has a k4-specific cost to ΔR (or seed luck — n=2). CONSEQUENCE:
  vec8 extra seeds must NOT be pooled with vec32 headline cells (confirmed);
  and the PLR arms (also vec8) must be compared against the vec8 uniform
  pair, which is the geometry-matched control.
- 2026-07-14: **METHODOLOGICAL: cross-arm ΔR comparisons need own-cell
  selectors.** Under the vec32 headline selector, vec8 runs show spurious
  negative ΔR (s46 −0.79(!), gap −0.09) because each policy's failure set
  differs; with own per-rollout selectors: s45 +0.56, s46 +0.12, gap +0.58.
  Report cross-arm tables with own selectors (+ fixed-selector appendix).

- 2026-07-12: **k5 convergence curve, 3 seeds × 4 checkpoints (kills
  "undertrained" definitively).** All k5 wids evaled at iters 90/180/270/365
  (~0.7/1.5/2.2/3.0B), fixed final-eval selector n=93,
  `outputs/eval540_k5progress/`. Seed-mean adaptable ΔR: +0.01 → +0.74 →
  +0.59 → +0.86. Reads: (1) adaptation emerges between 0.7B and 1.5B and
  PLATEAUS — not rising enough at 3B to rescue an "undertrained" story for
  the old flat cells (recipe artifact confirmed); (2) per-seed checkpoint
  trajectories are individually volatile (s42 dips to −0.02 at 2.2B while
  s43/s44 sit at +0.9; s43 is −1.10 at 0.7B then +1.46 at 3B) but the dips
  are idiosyncratic, NOT a systematic late erosion — do not over-read the
  single-seed zero-shot↔ΔR anticorrelation beyond what the entropy trade-off
  already establishes. Candidate appendix figure: mean ΔR vs training steps.
- 2026-07-12: **Task C seed 45 (kukzhob9, 0.10/k4) done + evaled** (iter 456,
  CSVs in eval540_return). Adaptable-map ΔR +0.37 — positive (cell now 4/4
  seeds positive) but below the original trio (+1.26/+0.71/+0.78): 4-seed
  +0.78 ± 0.36 vs 3-seed +0.92 ± 0.30. CAVEAT: s45 trained at vec 8/8/8 vs the
  originals' vec32, so it adds geometry variance, not pure seed variance —
  report the cell as 3-seed headline + s45/s46 as a same-recipe-smaller-batch
  robustness pair, don't silently pool. Seed 46 (ey3ia9c5, same cell) lands
  ~07-13. NOTE: I put both extra seeds on the 0.10 cell (handoff Task C
  wording was ambiguous between that and one per cell); 0.20-cell extra seeds
  remain optional follow-up.
- 2026-07-12: **k=5 ANOMALY RESOLVED — memsafe artifact, k5 adapts like k4
  (Task A complete).** Standard-recipe k5 (wids q924lklb/qyjag2qw/15lpuj3s,
  3B, iter 365, evals in `outputs/eval540_k5std/`): adaptable maps (own p0<0.8
  selector, n=93) t0..t4 = −2.18 → −1.32, **ΔR +0.86** (per-seed +0.55/+1.46/
  +0.56, all positive) vs old memsafe k5 cells ΔR +0.09 (n=118, dead flat) and
  k4 headline +0.92. Paper: k-sweep claims scope to standard-recipe cells;
  "k=4 sweet spot" softens to "adaptation robust for k≥3; k=2 degrades".
  Caveat: new k5 ran vec 8/8/8 (32 GB constraint) vs cluster vec32 — zero-shot
  all-540 t0 is lower (−0.03 vs k4's +0.24), so absolute cross-k levels stay
  geometry-tainted; the within-episode ΔR (locked metric) is the comparison.
- 2026-07-12: Wave 2 launched as wave-1 lanes finished: PLR **gap** arm
  qof5vici (GPU1), PLR **success** arm cnoj5ric (GPU0), both tag
  plr_map_curriculum_k4_e010, seed 42, 3B; Task C **seed 46** ey3ia9c5 (GPU2,
  grid tag). k4 s45 (kukzhob9) still training on GPU3 (~done midday).

- 2026-07-10: k5 lanes: epoch-10 in-training evals OOM non-fatally on 32 GB
  (training process's reserved allocator pool starves the separate-context
  eval subprocess). Runs CONTINUE fine; in-training eval curves will be
  missing for q924lklb/qyjag2qw/15lpuj3s — reconstruct offline from the
  every-10-epoch checkpoints if needed (offline eval is the locked metric
  anyway). Fixed for future runs: `torch.cuda.empty_cache()` before the eval
  subprocess spawn in pufferl.py.
- 2026-07-10: **h512 (k1w6dtm4) CRASHED at epoch-10 in-training eval and is
  DROPPED** — eval subprocess loaded the h512 checkpoint into an h256 model
  (size mismatch on positional_embedding), i.e. pitfall-#3 hidden-size
  propagation failed again on this box. Decision (user-endorsed): don't
  retry — the point was already geometry-confounded (vec4 vs the ablation's
  vec32, same confound class that invalidated the memsafe k5 cells) and the
  capacity curve 64→128→256 already reads "saturating"; paper footnotes
  "h512 exceeded the memory budget at comparable geometry". GPU3 reassigned
  to **Task C seed 45, 0.10/k4 standard recipe: wid kukzhob9** (tag
  ada_k4_gb3_legacy_eval_fix = original grid tag, vec 8/8/8, mb 8040).
- 2026-07-10: **TASK A + B TRAINING LAUNCHED on the vast box** (all 4 GPUs).
  k=5 standard recipe (tag `k5_standard_recipe_e010`, partner 2e029h15,
  e_ub=0.10, h=256, 3B steps): wids **q924lklb** (s42/GPU0), **qyjag2qw**
  (s43/GPU1), **15lpuj3s** (s44/GPU2). h=512 capacity point (tag
  `hidden_size_ablation_k4_e010`): wid **k1w6dtm4** (s42/GPU3).
  RTX-5090-fitted geometry after 4 smoke rounds: k5 = vec 8/8/8 +
  max_minibatch 10050 (~25.1 GB, ~18-22K SPS → 3B ≈ 40 h); h512 = vec 4/4/4 +
  max_minibatch 8040 (~21-30 GB, ~8K SPS → 3B ≈ 4.5 d). Constraints learned:
  max_minibatch must divide 50×horizon AND be a multiple of horizon; the h512
  OOM is resident-memory in rollout forward_eval (scales with vec, not
  minibatch) — vec16 and vec8 both OOM at h512, vec4 fits. NOTE for eval
  comparisons: vec geometry differs from cluster cells (batch ≈ 4.1M vs 26M
  tokens per epoch at k5) → epoch counts differ; per-trial return evals are
  unaffected.
- 2026-07-10: **HARD-UNSEEN (§0.3) WIDER PASS (hard25, 106 maps × 40
  rollouts) — TRANSFER IS WEAK AND SEED-INCONSISTENT.** Adaptable-unseen maps
  (p0<0.8): 0.10/k4 (n=18) ΔR +0.77 (split-half robust: +0.69/+0.81, 14/18
  maps improve, median +0.41) BUT per-seed +2.41/−0.05/−0.05 — one seed
  carries it; 0.20/k4 (n=15) ΔR −0.31 (per-seed +0.84/−0.81/−0.96), t1 dips
  below t0. All-106 zero-shot fine (+0.26/+0.30). Compare nuplan_hard where
  ALL seeds improve (+0.7…+1.7). Honest paper framing (appendix): in-context
  adaptation is robust on interaction-dense maps from the training
  distribution; on never-trained hard maps it is attenuated and
  seed-inconsistent at our scale — consistent with AdA needing large-scale
  task diversity for generalizable adaptation; our 5K-map pool is ~1/1000th.
  Main-text claims unaffected (phrased as adaptation, not held-out
  generalization, per settled risk #1). Data: outputs/eval540_heldout_hard25/.
- 2026-07-10: **HARD-UNSEEN (§0.3) FIRST PASS — NO CLEAR ADAPTATION TRANSFER,
  BUT UNDERPOWERED (n=7–8).** Built `nuplan_heldout_hard` (46 maps: ids ≥ 4999
  never trained AND sdc_interaction_steps ≥ 52 = exact nuplan_hard top-10 %
  cutoff; manifest in outputs/eval540_heldout_hard/). Headline cells, 20
  rollouts: 91 % of rollouts already succeed (subset saturated for the agent);
  p0<0.8 keeps only 8 (0.10/k4) / 7 (0.20/k4) maps. On those: ΔR +0.24 / −0.89
  (mean), 4/8 and 3/7 maps improve, per-seed spread −2.5…+1.8 — nothing
  interpretable at this n given known per-map cross-seed irreproducibility
  (risk #8). DO NOT cite as confirmation OR refutation yet. Follow-up in
  flight: `nuplan_heldout_hard25` (106 maps, threshold ≥ 21 = hardtrain top-25 %
  cutoff) × 40 rollouts × 6 wids for real power. If the null persists there,
  the honest appendix framing is: "in-context adaptation is robust on
  interaction-dense maps seen in training; transfer of the adaptation
  behavior to never-trained hard maps is not statistically detectable at our
  sample sizes" — and the main-text claims (already phrased as adaptation,
  not held-out generalization, per settled risk #1) need no change.
- 2026-07-10: **SPLIT-HALF SELECTOR ANALYSIS — SELECTION BIAS NEGLIGIBLE (§0.2,
  reviewer risk #2 SETTLED).** Six fresh headline-cell evals (540×20, local
  vast GPUs, ~22 min each) with new `--dump-per-rollout` raw records →
  `outputs/eval540_splithalf/`. Disjoint-half (select p0<0.8 on rollouts 0–9,
  measure ΔR on 10–19, and reverse): 0.10/k4 +0.94/+0.89 vs full-sample +1.00;
  0.20/k4 +1.24/+1.30 vs +1.37. Same-half biased references barely higher →
  regression-to-the-mean ≤ ~0.1 return units (~10 % of effect). Bonus: fresh
  evals replicate original headline ΔR within rollout noise (+1.00 vs +0.92;
  +1.37 vs +1.32) — second independent replication. Analysis:
  `scripts/adaptive/analyze_splithalf.py`.
- 2026-07-10: **HUMAN-NORMALIZED SCORES LANDED (§0.4)** — `scripts/adaptive/
  analyze_human_norm.py` → `outputs/eval540_norm/{human_norm_scores.csv,
  g_human_norm_curves.jpg}`. Denominator: per-map human-log return = t0 of
  demo-mode evals, byte-identical across all 6 demo wids (std=0 — policy-
  independent as claimed); saved to `outputs/eval540_demo/human_return_per_map.csv`.
  Maps with human_R < 0.5 excluded (31/539; 6 have human_R ≤ 0 — human log
  itself crashes). AdA-style median/20th-pct, seeds-averaged, headline cells:
  (a) ALL maps median ≈ 1.0 at every trial — the agent matches the human on the
  typical (saturated) map; adaptation is a TAIL phenomenon. (b) Adaptable maps
  median: 0.10/k4 −0.97→+0.13, 0.20/k4 −0.76→+0.04 — from "far below human" to
  "human-level median" within 4 trials. (c) 20th-pct climbs but stays well below
  human (−2.9→−1.4 / −4.6→−1.9): hardest tail not solved. (d) All-maps 20th-pct
  (0.10/k4): −0.10→+0.35 — AdA-Fig-4-shaped tail improvement without any selector.
  Candidate Fig 1 panels; (d) is selector-free and immune to reviewer risk #2.
- 2026-07-10: §0.2 split-half infrastructure: `--dump-per-rollout` flag added to
  `eval_final_540.py` (raw per-(map,rollout) success/return records; evaluator
  already had them in-memory). Verified rollout-mean reproduces per-map CSVs to
  6e-8. Six headline-cell re-evals with dumps launched locally
  (outputs/eval540_splithalf/, logs/splithalf_gpu*.log).
- 2026-07-10: **MEMORY-ABLATION CONTROL CONFIRMED — Fig 5 settled** (array
  13142075 w/ fixed per-agent trial-boundary reset, `outputs/eval540_cachereset2/`,
  fig g_memory_ablation.jpg). Per-trial return t0→t3, adaptable maps, mean±std
  over 3 seeds:
  0.10/k4 (n=108): intact −1.73→−0.81 (ΔR +0.92) vs reset −1.83→−1.76 (ΔR **+0.07**);
  0.20/k4 (n=62): intact −2.55→−1.23 (ΔR +1.32) vs reset −2.60→−2.68 (ΔR **−0.08**).
  Reset t0 ≈ intact t0 (same policy zero-shot — clean control), curve dead flat
  across trials → **cross-trial K/V memory is causally necessary for the entire
  adaptation effect.** Unlike the Jul-5 no-op, curves clearly diverge from intact.
  Recomputed independently from per-map CSVs on the vast box (matches the
  cluster-generated figure: +0.93/+0.08 & +1.32/−0.08; my selector n=108 vs
  figure's 106 — trivial selector-implementation difference, numbers unchanged).
  `verify_per_agent_reset.py` unit tests pass locally (scalar-vs-vector, reset-vs-
  fresh, other-agents-unaffected). Do not cite eval540_cachereset (v1, no-op);
  cite cachereset2 only.
- 2026-07-10: vast box (4×5090) is now the primary and only compute. Full data
  rsync from cluster landed: nuplan_201 (5402), nuplan_hard (540), heldout_403,
  outputs/ (9 GB), 6 headline checkpoint dirs (final iters 114/114/114/80/110/114).

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
