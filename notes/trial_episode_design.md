# Trial-as-episode redesign for adaptive driving

## Goal

Replace the current scenario-based episode structure with **variable-length trials packed into a fixed-budget episode**, where each trial is an independent "fresh map, fresh agent" goal-reach attempt and the K/V cache persists across trials within an episode.

## Decided semantics (from user)

1. **"New trial"**: fresh agent placement (back to start), fresh goal, fresh recorded-human trajectories restarting from t=0. **Same map** within an episode (no map-swap-per-trial). The map_rand_per_scenario / _reinit_envs_with_new_maps machinery is being removed entirely from the codebase as part of this work.
2. **Episode budget**: fixed at `k * scenario_length` ticks. If trials finish early, fill remaining budget with *another fresh episode* (start trial 0 of a new episode in the same buffer slot, separated by terminal flag).
3. **K/V cache persistence**:
   - **Adaptive (treatment)**: cache persists across trials within an episode. Only resets at episode boundary.
   - **Control**: cache resets at every trial boundary. Configurable knob.
4. **Full design, not MVP.** OK with 5-7 days of work.

## Code to be DELETED in this refactor

- `map_rand_per_scenario` flag and all its handling sites (drive.py, drive.h, eval scripts)
- `_reinit_envs_with_new_maps` Python function (drive.py:888-975) — entire function gone
- The `_render_keep_client_on_swap` / `vec_donate_client` / `vec_adopt_client` plumbing that exists *only* to keep raylib alive across the now-removed reinit
- The "broken as an ICL probe" warning block in adaptive.ini

The trial-boundary swap is now a fresh, smaller C function (`start_new_trial`) that does ONLY agent-state reset + goal sample, no env tear-down, no terminals write. New goal source: either `sample_new_goal()` (a road-lane point ahead of agent) or another agent's `init_goal` from the same map.

## Key implementation insight (from Plan agent)

`done_mask = terminals + truncations` controls cache reset (pufferl.py:610).
GAE only uses `terminals` for bootstrap truncation (not truncations).

So we have three distinct events:

| event | terminals | truncations | cache | GAE bootstrap |
|-------|----------:|------------:|:------|:--------------|
| within-trial step | 0 | 0 | continues | continues |
| **trial boundary (adaptive)** | 0 | 0 | continues | continues across |
| **trial boundary (control)** | 1 | 0 | resets | truncates |
| episode boundary | 1 | 0 | resets | truncates |

The episode boundary marker lets pufferl pack two episodes into one segment row — `create_episode_mask` will block cross-episode attention via the cumsum-on-terminals episode-id mechanism. Variable-length episodes inside a fixed-size segment work out of the box this way.

## Architecture

### C side (drive.h)

**New goal_behavior mode `GOAL_TRIAL = 3`**:
- On goal-reach: full goal_weight reward, mark `current_goal_reached=1`, set per-agent `trial_complete=1`. **Do not** stop, do not respawn (yet — Python decides).
- A new per-env counter `current_trial` (uses existing `current_scenario`-style logic).

**New per-env flag accessible from Python**: `trial_ended_this_step[env_idx]`. Set to 1 in the C step when:
- All ego agents in the env have reached goal, OR
- The trial-length budget (currently `scenario_length`) has elapsed without reach.

This flag is consumed by Python at the same tick — Python decides whether to swap maps, whether to set terminals, etc.

**No `_reinit_envs_with_new_maps` involvement.** Trial boundary uses a new lighter C function `start_new_trial(env, agent_idx)`:
- Reset agent position via `set_start_position` for that agent only
- Sample new goal via `sample_new_goal()` (or pick from another agent's `init_goal`)
- Reset per-agent metrics (`current_goal_reached=0, collided_before_goal=0, stopped=0, removed=0`)
- Reset agent's per-trial Log fields
- DO NOT reset `env->timestep` — the experts/partners continue along their recorded trajectory
- DO NOT touch other agents

Episode boundary (`tick == episode_budget`): pufferl handles this via the standard `done_mask` path. We set `terminals[ego_ids]=1` from Python, the env is fully reset by `binding.vec_reset` on the next `puffer_env.reset()` call. No special C function needed; the existing `c_reset` path does it.

**Per-trial Log fields** (in `Log` struct):
- `trials_completed_this_episode` (int)
- `trials_attempted_this_episode` (int)
- `mean_time_to_goal_per_trial` (float, running mean)
- `per_trial_succeeded[MAX_TRIALS]` (fixed-size array, MAX_TRIALS=8)
- `per_trial_collided_before_goal[MAX_TRIALS]` (fixed-size array)

vec_log aggregation accumulates these across envs.

### Python side (drive.py + adaptive.py)

**`AdaptiveDrivingAgent.__init__`**: add knobs
- `goal_behavior_trial = True` (whether to use new mode)
- `max_trials_per_episode` (default = `k_scenarios`)
- `trial_length` (default = `scenario_length`)
- `trial_cache_reset` (False = adaptive treatment, True = control)
- `episode_budget = max_trials_per_episode * trial_length` (just an alias)

**`Drive.step` modifications**:
- Per-step: read `trial_ended_this_step` array from C.
- For each env where trial ended:
  - Call C function `start_new_trial(env, ego_idx)` for each ego in that env
  - Increment `current_trial`
  - If `trial_cache_reset=True`: set `truncations[ego_ids_in_env]=1` (resets cache via done_mask, does NOT truncate GAE)
  - Aggregate trial metrics into per-trial dict
- After all per-env trial-ends processed:
  - Increment `episode_ticks` counter
  - If `episode_ticks >= episode_budget` OR `current_trial >= max_trials_per_episode`:
    - Set `terminals[ego_ids]=1` for ALL envs (episode boundary; pufferl's c_reset path handles the actual env reset on the next call)
    - Reset per-episode counters
    - Compute episode-level delta metrics (analogous to current `_compute_delta_metrics`)
    - Aggregate per-trial metrics list

**`_compute_delta_metrics` generalization**: now per-trial. Returns `trial_N_score` for each completed trial, plus `ada_delta_score = trial_(last)_score - trial_0_score`.

### Eval side (evaluator.py + utils.py)

**`HumanReplayEvaluator.rollout`**: rewrite outer loop to be trial-aware:
```python
for episode in range(num_episodes):
    obs, _ = env.reset()
    state = _fresh_state()
    trial_metrics_per_episode = []
    while True:
        for tick in range(self.sim_steps):
            obs, rew, dones, truncs, info = env.step(action)
            ...track per-trial success_arr...
            if dones.any():
                break  # episode ended
        else:
            continue  # trial ended but episode budget remaining
        break
    aggregate per-trial metrics
```

Per-`(rollout, agent, trial_idx)` success matrix instead of per-`(rollout, agent, scenario)`.

`RECOVERY_CACHE_RESET_PER_SCENARIO` → `RECOVERY_CACHE_RESET_PER_TRIAL` env var (control switch; ON = control, OFF = adaptive).

### Pufferl integration

Should require **zero changes**. The recv loop, GAE kernel, and `create_episode_mask` already handle:
- Multiple episodes per segment row (terminals=1 marks the joins)
- Cache reset on `done_mask=t+d`
- GAE truncation only on terminals

Verify with a synthetic test before assuming.

### Config knobs (adaptive.ini)

```ini
[env]
goal_behavior = 3                # GOAL_TRIAL mode
max_trials_per_episode = 4       # how many trials per episode max
trial_length = 100               # per-trial budget in ticks
episode_budget = 400             # max total episode ticks (= max_trials * trial_length)
trial_cache_reset = False        # adaptive treatment (cache persists). True = control.
```

## Implementation milestones

### Milestone 1 — C-side trial mode + Python signal (Day 1-2)

- Add `GOAL_TRIAL=3` constant + new branch in `c_step` goal-reach logic.
- Add `trial_ended_this_step[env]` field to env struct, exposed via a new binding (`vec_get_trial_ended` or similar) or via observation channel.
- Add per-trial Log fields.
- Unit test: instantiate one Drive env with goal_behavior=3, step until agent reaches goal, verify trial_ended flag fires, verify trial_complete count increments.

### Milestone 2 — Python wiring + map swap (Day 3)

- Refactor `_reinit_envs_with_new_maps` to take `set_terminals=True` param (default True for backwards compat).
- In `Drive.step`, read trial_ended array, call reinit-without-terminals at trial boundary.
- Increment current_trial, handle episode-budget exhaustion.
- Set terminals at episode boundary; integrate with map swap.
- Track per-trial metrics.

### Milestone 3 — `_compute_delta_metrics` rewrite (Day 4)

- Generalize from per-scenario to per-trial.
- Emit `trial_N_score`, `trial_N_collision_rate`, etc.
- Compute `ada_delta_score` as last_trial_score - first_trial_score.
- Add `mean_trial_score`, `n_trials_completed_per_episode`.

### Milestone 4 — Eval mirror (Day 5)

- Rewrite `HumanReplayEvaluator.rollout` outer loop.
- Per-(rollout, agent, trial) success array.
- Mirror trial_cache_reset for control runs.
- Test on a saved checkpoint.

### Milestone 5 — Integration test + verification (Day 6)

- End-to-end: train one epoch with new mode, check loss not NaN, scores look sensible.
- Compare GAE flow with `terminals=1` only at episode boundary vs at every trial boundary; verify expected behavior.
- Log `transformer_position` over an episode to verify cache lifecycle.

### Milestone 6 — A/B run + ship (Day 7-10)

- Train 3 seeds k=2 (now 4 trials × 100 ticks per trial = 400-tick episode budget) gb=3 — adaptive treatment.
- Train 3 seeds k=2 — control (trial_cache_reset=True).
- Compare ada_delta_score curves.

## Risks / open questions

1. **Cost of map-swap-per-trial**: `_reinit_envs_with_new_maps` is the slow call (vec_close + 540× env_init). Per-trial cost ~5-10s. With 4 trials/episode and many parallel envs, this could 4× training wall-clock.
   - Mitigation: could pre-load N maps and just switch index, but env doesn't support that today. Adds engineering.

2. **K/V cache size**: with episode_budget=400 and horizon=400, cache exactly fits. With variable trial counts, the cache might overrun if trials run long. We need to cap episode at horizon to avoid wraparound.

3. **Per-trial Log fixed-size arrays**: MAX_TRIALS=8 is arbitrary. If a user sets max_trials_per_episode=16, breaks. Needs runtime validation.

4. **Per-env trial counters when num_envs=540**: 540 separate counters. Cheap in C, just need to expose them right.

5. **`map_rand_per_scenario` flag**: existing knob is "broken as ICL probe" because of the terminals=1 write. Our new mode bypasses this. Should we also fix the existing flag, or remove it / mark deprecated?

## Smallest-possible-test before committing to full impl

(Removed — Day 0 used `map_rand_per_scenario`, which we're deleting. Going straight to the full implementation.)
