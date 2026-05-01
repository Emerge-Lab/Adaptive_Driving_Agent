# Why doesn't the adaptive ego use its K/V cache? (and the oracle test plan)

Investigation date: 2026-05-01.
Author: Claude (working with Mohit on the adaptive-driving paper).

---

## Empirical setup that triggered this investigation

We trained two adaptive ego policies against the entropy-sweep partner
`6rauydj2` (trained on `entropy_ub ∈ [0, 0.5]`):

- **`curr_e0.5`** (wandb `hprfn8dc`) — entropy-curriculum on (ub
  annealed 0.025 → 0.10 → 0.25 → 0.50 across the first ~120 episodes).
- **`nocurr_e0.5`** (wandb `7wm1sk5v`) — entropy-curriculum off
  (ub fixed at 0.50 throughout).

Both trained from scratch to ~epoch 203 (resumed from intermediate
checkpoints after host OOMs; resume restores optimizer + global_step
cleanly via `pufferl.py:280`). Same partner, same seed, same map dir,
same rollout config — **the only varying factor is the curriculum**.

The training-time `ada_delta_score` favored `curr` over `nocurr`. We
ran three offline analyses on the final checkpoints to understand
**why** and **whether the policy is actually using the cache for
adaptation**.

---

## Three offline analyses (all on 300 truly-held-out scenes)

Held-out set: `resources/drive/binaries/nuplan_201_heldout300/` —
symlinked `map_001..300` → originals `map_5102..5401` (training used
the first 4999, so these are unseen).

### Analysis 1: head-to-head eval (`puffer eval`, no co-player)

```
metric                              nocurr_e05    curr_e05      delta
─────────────────────────────────────────────────────────────────────
s0_rate                                 0.5906       0.6913    +0.1007
s1_rate                                 0.5906       0.6946    +0.1040
ada_delta                               0.0000       0.0034    +0.0034
p_s1_given_s0_pass                      0.9830       0.9854    +0.0025
p_s1_given_s0_fail                      0.0246       0.0435    +0.0189
n_s0_pass                                  176          206       +30
n_s0_fail                                  122           92       -30
```

Curr is ~10 pp better in absolute success rate in BOTH scenarios on
held-out scenes. ada_delta is ~0 for both — the curriculum gain is a
*better policy* gain, not an *adaptation* gain. Recovery rate
(`p_s1_given_s0_fail`) almost doubled with curriculum (2.5% → 4.4%) but
both numbers are tiny in absolute terms.

### Analysis 2: attention probe (`scripts/probe_attention.py`)

Cross-scenario attention mass = fraction of attention from s_1 query
positions to s_0 cache slots. Recorded per (layer, head) over a single
rollout against `6rauydj2`.

| Layer/Head | curr_e05 | nocurr_e05 |
|------------|----------|------------|
| L0H0       | 0.44     | 0.68       |
| L0H1       | 0.51     | 0.69       |
| L0H2       | 0.55     | 0.67       |
| L0H3       | 0.49     | 0.67       |
| L1H0       | **0.14** | 0.66       |
| L1H1       | 0.35     | 0.70       |
| L1H2       | **0.05** | 0.63       |
| L1H3       | 0.23     | 0.65       |

`curr` attends LESS to past than `nocurr` across every head — opposite
of the cache-use hypothesis.

### Analysis 3: counterfactual cache (`scripts/counterfactual_cache.py`)

Paired rollouts: same map, same seed, same partner conditioning.
Condition A preserves the K/V cache from s_0 → s_1; Condition B zeros
it. Difference isolates "does cache CONTENT matter?"

| Metric                              | curr_e05 (z) | nocurr_e05 (z) |
|-------------------------------------|--------------|----------------|
| Paired s1 lift (preserved − zeroed) | -0.007 (-0.45) | +0.030 (+1.41) |
| P(s1 \| s0 fail) preserved          | 0.30         | 0.14           |
| P(s1 \| s0 fail) zeroed             | 0.30         | 0.10           |

`curr` is unaffected by cache zeroing (z = -0.45). `nocurr` does feel
it — recovery rate drops ~30% relative when cache is wiped.

### Joint reading

| Question                          | curr_e0.5         | nocurr_e0.5  |
|-----------------------------------|-------------------|--------------|
| Better single-shot driver?        | Yes (69% vs 59%)  | No           |
| Attends to past more?             | No (0.05–0.55)    | Yes (0.63+)  |
| Cache content load-bearing?       | No (z = -0.45)    | Marginally yes (z = +1.41) |

The curriculum produced a stronger driver that **ignores** the cache.
The non-curriculum policy attends and (slightly) uses the cache, but
that doesn't translate into better absolute scores.

**Either way, neither policy has meaningful in-context adaptation.**

---

## First-principles teardown: what does adaptation actually require?

For the ego to "adapt to partner P across scenarios" via cache, ALL of
these must hold:

1. **Partner type is observable** — some signal in ego's obs encodes
   partner type.
2. **The encoder preserves it** — the encoder doesn't lose the
   discriminating info during compression.
3. **The hidden states get written to cache** — automatic given the
   architecture.
4. **The policy queries the cache** — attention pattern looks back.
5. **The retrieved values still carry the info** — what's stored is
   still partner-relevant.
6. **The policy conditions on cache content** — the policy learns to
   change its action distribution based on cache values.

We have evidence on each link:

- (1) **Weak**: partner conditioning is NOT in ego obs. The ego sees
  the partner's `(rel_x, rel_y, rel_heading_x, rel_heading_y, width,
  length, rel_signed_speed)` — instantaneous behavior only. To infer
  partner type (e.g., `entropy_weight`), the ego must integrate
  *behavioral variance* over many timesteps. From a single snapshot,
  partner type is unobservable.
- (2) **Suspicious**: encoder compresses 1855 → 256. Partner-typing
  info needs to survive this bottleneck, but the only push for it is
  RL gradient — which is weak unless the policy is already using
  partner type info.
- (3) ✓
- (4) ✓ (probe confirmed)
- (5) **Likely the break**: counterfactual cache shows zeroing the
  cache barely changes s_1 — the values don't carry usable info.
- (6) **The break**: if the values aren't useful, conditioning on
  them can't help.

### Three specific architectural concerns

**(a) Observations are in EGO FRAME**
Every partner feature (`rel_x`, `rel_heading_x`, `rel_signed_speed`)
is computed relative to the ego's CURRENT pose. As the ego moves, the
same world-frame partner behavior produces different obs values. The
K/V cache stores ego-frame snapshots, but the ego frame is
non-stationary across scenarios. To compare partner behavior across
the scenario boundary, the policy would have to implicitly invert the
ego frame change — a hard sub-problem.

**(b) Partner identity is unlabeled**
Slot K of partner_obs is "the K-th nearby agent in
`active_agent_indices` iteration order." If 60 partners are within
visibility, slot 5 could be partner-id-42 at one step and same agent
at the next (stable for stationary scenes), but there's no identity
TAG. Comparing slot K values across timesteps requires the policy to
verify that's the same partner — an implicit sub-problem.

**(c) The latent we want to identify is BEHAVIORAL VARIANCE**
A high-entropy partner samples actions stochastically. Its
instantaneous position+speed look identical to a low-entropy partner.
The DIFFERENCE only shows up as "the partner did something
unexpected" — which requires the policy to have a forward model of
what the partner *should* be doing under each conditioning value, then
notice deviations. That's an enormous hidden inference problem, and
nothing pushes the policy to learn the forward model.

### Conclusion

End-to-end, we're asking the policy to learn (from sparse goal-reach
RL reward) to:
1. Track partner identities across timesteps (no ID labels)
2. Undo ego-frame motion to recover world-frame trajectories
3. Estimate behavioral variance per partner over many timesteps
4. Map variance → partner-type embedding
5. Use that embedding to modulate actions

That's a tower of hard implicit inference problems on top of "drive a
car well." The optimization just doesn't push hard enough for steps
1-4, especially when most of the reward can be earned by single-shot
driving alone. This explains the empirical finding: the better
single-shot driver (curr) ignores the cache, because the gradient
toward "be a better driver" is much stronger than the gradient toward
"learn to read the cache."

---

## What's already in the codebase that could help

`grep -rin oracle …` returns NO hits anywhere in `/workspace/ADA`.

The closest thing is the `[env.conditioning]` section in
`adaptive.ini` — but this is **per-ego self-conditioning** (each ego
agent gets its OWN conditioning vector that determines its OWN reward
function: `reward_collision_weight`, `entropy_weight` for its own
exploration, etc.). It is plumbed end-to-end:

| Layer              | What happens                                     |
|--------------------|--------------------------------------------------|
| Python `Drive.__init__` | kwarg `conditioning={}` → `self.entropy_conditioned`, etc. (drive.py:60-167) |
| Adds `conditioning_dims` to `self.ego_features` so `obs_dim` accounts for the slots (drive.py:177)  |
| C `env->collision_weights[i]` etc. allocated per active ego agent (drive.h:1841-1844) |
| C samples weights at episode reset (drive.h:2486-2494) |
| C appends weights to each ego's obs in `compute_observations` (drive.h:2294-2302) |

This is NOT what we want, but the slots exist and are reachable. We
can **hijack them**: keep the obs format the same, but at sample time,
overwrite the per-ego conditioning weights with this env's partner
conditioning vector. Each ego in env E sees the same vector its
partner is using.

---

## Plan: oracle test (give ego the partner's conditioning)

### Goal

A clean diagnostic: if the ego is just *handed* the partner's
conditioning vector as obs, does the policy then learn to adapt
(drive differently when entropy_weight is high vs low)?

- **If yes**: the architecture downstream is fine. The bottleneck is
  inference from behavior. Then we can pursue real fixes (per-partner
  history features, world-frame obs, partner-attention layer, etc.).
- **If no**: the bottleneck is downstream — even with explicit
  partner-type signal, the policy can't condition action on it. Then
  we need to look at the policy head, the action distribution, or
  the loss formulation.

### C-side dive: where do the existing conditioning slots come from?

The per-ego conditioning weights `env->{collision,offroad,goal,entropy,discount}_weights[i]` are used in **two places** in `drive.h`:

1. **obs append** (lines 2294-2302) — appended to ego obs after `base_ego_dim`.
2. **reward computation** (lines 2625, 2637, 2669, 2679, 2691) — the per-step
   reward at collision / offroad / goal events SCALES with these weights.

This means we **cannot just enable `[env.conditioning].type=all` and
overwrite the C arrays** — that would also change the ego's reward
function (e.g., sampling collision_weight ∈ [-1, 0] makes some
collisions free).

### Pufferl-side dive: more silent uses of these slots

Even worse, **pufferl reads two of these slots from obs to drive
training dynamics**:

| Slot | Read by | Becomes |
|------|---------|---------|
| collision_w, offroad_w, goal_w | (only C-side reward) | per-agent reward scaling |
| entropy_w (slot at offset 12 with reward+ego) | `pufferl.py:885` | per-agent entropy weight in PPO loss |
| discount_w (slot at offset 13 with reward+ego) | `pufferl.py:743` | per-agent γ in GAE advantage computation |

The `pufferl` hooks are gated on `entropy_conditioned` /
`discount_conditioned` flags on the env. If we naively enable
`[env.conditioning].type=all` to get the obs slots and then override
them in Python, pufferl will use the **partner's** entropy/discount
values as the **ego's** PPO hyperparameters — a major silent training
distortion.

### The complete "no behavior change" recipe

To get the obs slots while preserving the no-conditioning training
dynamics:

1. **Force `[env.conditioning].type = "all"`** in the env init (from
   the `ego_is_oracle` flag), so 5 obs slots are allocated and the C
   `env->*_weights[i]` arrays exist.
2. **Pin all 5 sampling ranges to lb=ub=default** so the C-side
   per-agent samples are constant and equal to the default reward
   weights. Reward computation is identical to a non-conditioned run.
3. **After C-side setup, in Python, flip
   `self.entropy_conditioned = False` and
   `self.discount_conditioned = False`** (keep `reward_conditioned =
   True` — it has no pufferl-side hook). Pufferl now sees False on
   these flags and skips the per-agent γ / entropy hooks → uses
   global defaults.
4. **In step() / reset(), Python overrides the 5 obs slots in each
   ego row with `env_conditioning[env_of_ego]`** — the partner's
   conditioning vector. This is pure obs signal: pufferl no longer
   reads these slots, the policy just sees them as input.
5. **Bypass the `line 350` `NotImplementedError`** that forbids dual
   ego+co-player conditioning. The check predates oracle and is not
   protecting any actual coupling for our setup.

This makes the diff one-flag, no-C, contained:
- One env kwarg + ini knob (`ego_is_oracle`)
- Init-time validation + flag fix-up + bypass
- One step()-time obs override loop
- One sentence in the launcher to set the deterministic-default lb/ub on the 5 dims

### Implementation (Python obs override, no C changes)

1. **New env kwarg** `ego_is_oracle = False` (default off).
2. **Adaptive.ini knob** `ego_is_oracle = False` so CLI auto-generates
   the `--env.ego-is-oracle` flag.
3. **In `step()`** (drive.py, after `binding.vec_step(self.c_envs)`,
   before return), if `ego_is_oracle` is on AND `env_conditioning`
   has slots:
   - For each env e, for each ego in env e, overwrite the
     `[base_ego_dim : base_ego_dim + n_partner_dims]` slice of
     `self.observations[ego_id]` with `self.env_conditioning[e]`.
   - Same for the post-reset obs (need to call after `_set_env_variables` too).
4. **In `_add_co_player_conditioning`-style helper**: factor the
   slot-locating logic so we can reuse it for ego.
5. **Launcher**: must set `--env.conditioning.type all` AND
   `--env.conditioning.{collision,offroad,goal,entropy,discount}-weight-{lb,ub}`
   to default values (so the C-side reward isn't changed):
   ```
   --env.conditioning.collision-weight-lb -0.5  --env.conditioning.collision-weight-ub -0.5
   --env.conditioning.offroad-weight-lb  -0.5  --env.conditioning.offroad-weight-ub  -0.5
   --env.conditioning.goal-weight-lb      1.0  --env.conditioning.goal-weight-ub      1.0
   --env.conditioning.entropy-weight-lb 0.001  --env.conditioning.entropy-weight-ub 0.001
   --env.conditioning.discount-weight-lb 0.98  --env.conditioning.discount-weight-ub 0.98
   ```
   These are the same defaults as the no-conditioning case, so reward
   semantics are identical to a non-conditioned run. The slots in obs
   are then allocated but their values are overwritten by Python with
   the partner's per-env conditioning vector.

### Caveats / things to verify

- **Slot offset & width**: ego's conditioning width = 3 (reward) + 1
  (entropy) + 1 (discount) = 5 when type="all". Partner's width must
  match — also "all" → 5. We assert this at init.
- **The partner conditioning sample is per-env, valid for the whole
  episode.** Need to inject the same vector EVERY step (overwrite obs
  slots in step(), not just at reset).
- **The model checkpoint obs_dim with type=all is 1855** (ego_dim=14
  = base 9 + cond 5, partner_obs and road_obs unchanged). Fresh
  training run from scratch. Cannot load the curr/nocurr e=0.5
  checkpoints into this — their obs_dim is also 1855 but their slots
  contained their own conditioning, not the partner's; the policy
  weights have learned an interpretation that doesn't transfer.
  Train fresh.

### What success looks like

If the oracle test works, we expect to see — vs the no-oracle baseline:
- ada_delta improves significantly (e.g., +0.05+ instead of ~0)
- Per-condition behavior visible: rollouts at extreme partner
  conditioning (e.g., entropy=0.5 vs entropy=0.0) should produce
  visibly different ego policies (different distance kept from
  partner, different speeds, etc.)
- Counterfactual: zeroing the partner conditioning slots (not the
  full cache, just the oracle slots) should hurt s_0 score because
  the ego is now driving "blind" to partner type

### Run plan once implemented

User decision: **only do the curr experiment** — single oracle run, no
nocurr-baseline pairing. The comparison is against the existing
`curr_e0.5` (`hprfn8dc`) which has identical config except no oracle.

1. **Wire the flag** (drive.py + adaptive.ini).
2. **Smoke test** — first 30 sec of a launch, confirm:
   - obs slot layout is correct
   - the partner conditioning is actually being injected (print first
     few obs slot values, verify they match `env_conditioning[e]`)
   - training doesn't crash
3. **Full oracle run** — same config as `curr_e0.5`:
   - Partner: `6rauydj2` at `e_ub=0.5`
   - k=2/201, horizon=402, nw=24 nv=24 (or 32/32 if k_eff has freed
     RAM by then)
   - Entropy curriculum: ON (mirror `curr_e0.5`)
   - Ego conditioning: type=all with default-value lb=ub
   - Ego oracle: ON
4. **Analyses identical to the curr/nocurr runs**:
   - Eval on 300 held-out scenes (same `nuplan_201_heldout300`
     symlinked dir we already built)
   - Counterfactual cache (zero the cache to see if it matters now
     that the policy has the oracle slot)
   - Attention probe (does attention still go to past, or does the
     policy ignore the past now that oracle gives the answer?)

If the oracle policy adapts strongly (`ada_delta` jumps from ~0 to
0.05+, attention to past drops, conditioning slot is load-bearing),
the bottleneck is **inference from behavior** and we move on to fixes
like per-partner history features. If the oracle policy STILL doesn't
adapt, the bottleneck is **downstream** (action conditioning) and we
need to look at the policy head, action distribution, or loss.
