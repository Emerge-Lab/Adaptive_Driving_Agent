# Adaptive Population Play — paper experiment plan

40-hour execution window.  8 GPUs.  Each ego training ≈12h, each
co-player training ≈4.5h, each hard-map eval ≈4 min (M=10 × 540 maps).
Budget = 320 GPU-h; T1 ≈ 180, T1+T2 ≈ 300.

## What we already have

- **Adaptive ego baseline** `4lm6kkh7` (γ=0.995, partner `2e029h15`,
  e_ub=0.10) → `ada_delta = +0.222 ± 0.18` on `nuplan_hard` (200 rollouts).
- **5 partner-sweep adaptive egos** (one per entropy partner, γ=0.995,
  lane=0.025) — final ckpts at epoch 152.  Hard-map eval (M=10):

  | partner | e_ub | ada_delta ± std |
  |---------|-----:|----------------:|
  | miku2puk | 0.05 | **+0.155 ± 0.008** |
  | 2e029h15 | 0.10 | **+0.154 ± 0.085** |
  | m2ygolog | 0.20 | -0.010 ± 0.016 |
  | 6rauydj2 | 0.50 | -0.001 ± 0.008 |
  | n48teqjs | 1.00 | -0.000 ± 0.003 |

- **15-partner DE-grid in progress** on GPUs 0-4, ETA 22:10 UTC.
  Grid: `discount_lb ∈ {0.4, 0.6, 0.8} × entropy_ub ∈ {0.001, 0.01, 0.05,
  0.1, 0.2}`.  Wandb: `ada_coplayer_sweep`.
- **`nuplan_hard` map split** (top 10% by SDC interaction), recipe
  recorded in `notes/nuplan_hard.md`.
- **Fast eval recipe**: M=10, vanilla code, 5390 (map, ego) datapoints/eval,
  SE on mean ≈0.003.

## Key finding driving the plan

**Adaptation only emerges against deterministic partners (e ≤ 0.10).**
For e ≥ 0.20 the gap collapses.  This means:

- Robustness experiments need to vary the partner distribution along the
  *deterministic ↔ noisy* axis, not blindly use [0, 1].
- The "interesting" ada_delta lives in a narrow slice of partner space.
- We should structure the partner-distribution test to span that slice
  cleanly.

---

## Tier 1 — must-have for paper

### T1.1  Main result table: 4 ablation conditions vs adaptive-conditioned baseline

Using a single fixed partner (`2e029h15`, e=0.10 — matches the
`4lm6kkh7` baseline that gave +0.222), train 4 ego variants:

| label | description | training change |
|-------|-------------|----------------|
| **A** | Adaptive + conditioned partner | (baseline, already trained) |
| **B** | Non-adaptive + conditioned partner | reset ego K/V at every scenario boundary (`RECOVERY_CACHE_RESET_PER_SCENARIO=1` env var, or k_scenarios=1) |
| **C** | Adaptive + fixed-policy partner | partner = single ckpt, ego never sees conditioning slot |
| **D** | Self-play | both ego and "partner" use the same current policy |
| **E** | Adaptive + log-replay only | no learned partner, others follow recorded humans |

Cost: 4 trainings × 12h = **48 GPU-h**.  Eval: 5 × 4 min on free GPUs.

### T1.2  Scaling along k_scenarios

Train adaptive egos at `k_scenarios ∈ {1, 2, 4}`.  k=8 is 48h on its own
and probably won't fit; defer.

Cost: 12 + 12 + 24 = **48 GPU-h**.

### T1.3  Robustness: in-distribution vs OOD partner distribution

`[TBD — confirm exact ranges]`. Updated based on what we learned about partner space:

- **In-dist baseline**: train ego sampling partner conditioning over
  `e ∈ [0, 0.20]` (the slice where adaptation happens), test on same range.
- **OOD test**: train on `e ∈ [0, 0.05]` (very deterministic only), test
  on `e ∈ [0.10, 0.20]`.  Asks: does an ego trained on near-deterministic
  partners still show ada_delta when faced with a slightly noisier one?
- **Aggressive vs cautious test**: condition partners with
  collision_weight_lb=-2 (cautious) vs 0 (aggressive); reuse existing egos.

Cost: 1 new training (the OOD ego) + 2 evals = **12 GPU-h** + **20 min**.

---

## Tier 2 — should-have

### T2.1  Robustness: held-out maps

Eval-only.  Build a `nuplan_hard_holdout` split (different 10% of the
nuplan_201 maps that we did NOT score / use for `nuplan_hard`).  Eval the
5 partner-sweep egos against this set.  Cost: ~30 min.

### T2.2  Adversarial co-player

Train a partner with `collision_weight = +2` (rewards collisions); eval
all egos against it.

Cost: 4.5h partner + 5 evals × 4 min = **5 GPU-h**.

### T2.3  Scaling: number of training maps

Train ego on `nuplan_201` subsets of size 200, 1000, 5000.

Cost: 3 × 12h = **36 GPU-h**.  Defer if tight.

### T2.4  Conditioning leave-one-out (5 dims)

Train 5 egos, each with one conditioning dim disabled
(collision / offroad / goal / entropy / discount).

Cost: 5 × 12h = **60 GPU-h**.  Pick 2-3 dims if budget tight.

---

## Tier 3 — defer to post-paper unless time

- LSTM vs Transformer (one extra training)
- RNN state size sweep
- Co-player ratio (25 / 50 / 75 %)
- Diversity of conditioning ranges (wide vs narrow)

---

## Proposed 40h schedule

Times relative to the 09:55 UTC start of this plan.  All wall-clock
estimates assume 12h per ego training and 4.5h per co-player.

### Phase 1 (h0 → h13, overlaps with DE-grid finishing on GPUs 0-4)

Free GPUs: 5, 6, 7.  Launch 3 of the 4 T1.1 ablations:

| GPU | run | duration |
|----:|-----|---------:|
| 5 | T1.1-B  non-adaptive ego vs `2e029h15` | 12h |
| 6 | T1.1-C  adaptive ego vs fixed partner | 12h |
| 7 | T1.1-D  self-play | 12h |

### Phase 2 (h13 → h25, all 8 GPUs free)

DE-grid done at h~12.  Launch:

| GPU | run | duration |
|----:|-----|---------:|
| 0 | T1.1-E  log-replay ego | 12h |
| 1 | T1.2  scaling k=1 | ~6h |
| 2 | T1.2  scaling k=4 | ~24h (runs into Phase 3) |
| 3 | T1.3  OOD ego (train on e ∈ [0, 0.05]) | 12h |
| 4 | T2.2  adversarial partner (4.5h) → ego | ~16h |
| 5 | T2.4  leave-one-out: discount | 12h |
| 6 | T2.4  leave-one-out: entropy | 12h |
| 7 | T2.4  leave-one-out: collision | 12h |

Run T1.2 k=2 already exists in 4lm6kkh7; reuse for the curve.

### Phase 3 (h25 → h40, finishing + evals)

- k=4 finishes around h37
- All other Phase 2 trainings done by h25
- Free GPUs run evals (M=10 × 540 maps × ~4 min each)
- Build all paper plots:
  - **Fig 1**: main result bar plot (5 conditions, ada_delta ± std)
  - **Fig 2**: ada_delta vs partner entropy (uses 5 partner egos +
    DE-grid eval)
  - **Fig 3**: ada_delta vs k_scenarios
  - **Fig 4**: in-dist vs OOD partner distribution
  - **Fig 5**: leave-one-out conditioning ablation

---

## Open questions to resolve before launching

1. **Self-play definition (T1.1-D)**: ego = partner = same live policy
   on each step?  Or ego = current policy, partner = past-snapshot?  The
   former is "true" self-play but the partner is non-stationary; the
   latter is more like league play.  Pick one.
2. **Non-adaptive definition (T1.1-B)**: do we set `k_scenarios = 1` (so
   training never gives the ego cross-scenario context) or keep
   `k_scenarios = 2` and reset the cache at the boundary?  The latter
   isolates "is it the cache that helps?" cleanly; the former is more
   honest as a "non-adaptive" baseline.  Pick one or do both.
3. **OOD ranges (T1.3)**: my proposal `[0, 0.05]` train → `[0.10, 0.20]`
   test is one option.  Confirm this is the slice you want — alternatives
   are deterministic→noisy (`[0, 0.1]` → `[0.5, 1.0]`) which we already
   know breaks, or wide→narrow (`[0, 1]` → `[0.05, 0.2]`).
4. **Drop k=8?**  Yes unless you have a strong opinion.  48h alone.

Answer 1-4 and I'll start writing the launchers.

---

## Quick links

- DE-grid sweep status: `tmux attach -t coplayer_de_grid`,
  `/tmp/coplayer_de_grid_driver.log`
- Hard-map eval recipe: `notes/nuplan_hard.md`
- Latest ego eval results: `/tmp/eval_partner_sweep_m10.log`
- 5 partner-sweep ckpts:
  `experiments/puffer_adaptive_drive_{0bmlyvg3,l7c13x1m,0yih6s9k,sa80qcs2,p6q2b2xp}/model_..._000152.pt`
- Adaptive launcher template: `scripts/adaptive/nuplan_transformer_local_k2_201_partner_sweep.sh`
