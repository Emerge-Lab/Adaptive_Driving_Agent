# Trial Mode (`goal_behavior=3`)

Design and contract for the in-context adaptation training mode.

## Why

The adaptive ego is a Transformer with a KV cache. We want it to **adapt
across attempts within a single fixed-budget episode** — i.e., use what it
saw in trial 1 to do better in trial 2, etc. That requires:

1. Multiple goal-reach attempts ("trials") inside one episode.
2. KV cache that **persists across trial boundaries** (so context is
   preserved) but **resets at episode boundaries** (so episodes are i.i.d.).
3. PPO/GAE that **stops bootstrap at trial boundaries** (because the
   agent's value at $t+1$ is computed post-respawn, from a different
   state, and bootstrapping it into the last step of the old trial
   contaminates the target).

These three things — cache reset, GAE bootstrap-stop, episode-vs-trial
distinction — have different gates. The next sections specify each.

## Terms

| Term | Meaning |
|---|---|
| **Trial** | One goal-reach attempt. Ends on goal-reach OR `per_trial_timeout` ticks. |
| **Episode** | A sequence of at most `max_trials_per_episode` trials, sharing a single KV cache. |
| **Scenario** | A map. Under `goal_behavior=3`, each episode runs on **one** map (no per-trial map swap). |
| **`terminals[t]`** | 1 ⇔ the *episode* ended at step $t$. Used for both **cache reset** and **GAE bootstrap-stop**. |
| **`truncations[t]`** | 1 ⇔ a *trial* ended at step $t$ but the episode continues. Used **only for GAE bootstrap-stop**; cache persists. |
| **`trial_ended_this_step[i]`** | Per-agent C-side flag, set every trial boundary (goal-reach or timeout). Mirrored to `truncations` by Python. |
| **Cache reset** | Zero out the Transformer's K/V tensors. Done at episode boundary only. |
| **GAE bootstrap-stop** | Setting $(1-\text{stop}_t) = 0$ in the GAE recursion to prevent $V_{t+1}$ contamination across the boundary. |

## The two-boundary problem

Standard PPO has one boundary signal (`dones`). We need two, because the
two distinct things that happen at trial-vs-episode boundaries don't
align:

| Event | `terminals` | `truncations` | KV cache | GAE bootstrap |
|---|:-:|:-:|:-:|:-:|
| Within-trial step | 0 | 0 | continues | continues |
| **Trial end** (goal or timeout), more trials to go | 0 | **1** | **continues** | **stops** |
| **Episode end** (last trial done) | **1** | 0 | **resets** | stops |
| Scenario boundary (gb≠3 only) | 0 | 0 | n/a here | n/a |

Mnemonic: **`terminals` ⇒ cache reset; (`terminals` OR `truncations`) ⇒ bootstrap stop.**

## State machine — per agent, per `c_step`

```
                       ┌──────────────────────┐
                       │  agent.removed == 1  │ ───── skip (Option D)
                       └──────────┬───────────┘
                                  │ no
                                  ▼
       ┌─────── trial_ended? ─────┴───── neither ───→  continue trial
       │   (reached || timed_out)
       │
       ▼
  trial_ended_this_step[i] = 1
  trial_count++
       │
       ├── trial_count >= max_trials_per_episode  ───→ EPISODE END
       │       │
       │       ▼
       │   terminals[i]      = 1          // Python: cache reset HERE
       │   add_log_one_agent(env, i)      // flush this agent's metrics
       │   agent.removed     = 1          // Option D: idle
       │   agent.x, agent.y  = INVALID    // off-grid
       │
       └── otherwise  ─────────────────────────→ TRIAL END
               │
               ▼
           respawn_agent(env, i)              // back to start
           agent.respawn_timestep = -1        // clear ghost flag (see "Render gates")
           agent.trial_start_timestep = env->timestep
```

The Python side mirrors `trial_ended_this_step → truncations` after every
`vec_step`. So:

* Trial-end branch → C sets `trial_ended_this_step[i] = 1`.  Python sets
  `truncations[i] = 1`.  `terminals[i] = 0` (it was zeroed at the top of
  `step`).
* Episode-end branch → C sets BOTH `trial_ended_this_step[i] = 1` AND
  `terminals[i] = 1`.  Python sets `truncations[i] = 1`.

Both signals fire at the last trial end. That's intentional — the cache
reset gate (terminals) and the bootstrap-stop gate (terminals OR
truncations) both want to fire there.

## PPO / GAE formulation

Standard GAE (Schulman et al. 2016) with a single `done` signal:

$$
\delta_t = r_t + \gamma\,(1-d_t)\,V_{t+1} - V_t
$$

$$
\hat A_t = \delta_t + \gamma\lambda\,(1-d_t)\,\hat A_{t+1}
$$

In vanilla PPO, $d_t = \text{terminals}_t$. The $(1-d_t)$ factors zero out
the $V_{t+1}$ bootstrap and the recursive advantage at episode boundaries
(where state $t+1$ is a fresh env reset — no semantic relation to state
$t$).

**Trial-mode modification.** At every trial boundary (not just episode
boundary), state $t+1$ is the post-respawn state — back at the trajectory
start position with reset velocity. $V_{t+1}$ from that state is **not** a
valid bootstrap for state $t$ (the last step of the old trial, somewhere
else in the map). We define:

$$
\text{bootstrap\_stop}_t \;=\; \min\!\bigl(\text{terminals}_t + \text{truncations}_t,\; 1\bigr)
$$

and replace $d_t$ in BOTH GAE equations:

$$
\delta_t = r_t + \gamma\,(1-\text{bootstrap\_stop}_t)\,V_{t+1} - V_t
$$

$$
\hat A_t = \delta_t + \gamma\lambda\,(1-\text{bootstrap\_stop}_t)\,\hat A_{t+1}
$$

This is `pufferlib/pufferl.py`'s `bootstrap_stop = (self.terminals + self.truncations).clamp(max=1.0)`.

The KV cache reset is **independent**: it gates on `terminals` alone, NOT
on `bootstrap_stop`. Otherwise we'd lose the cross-trial context that is
the entire point of trial mode.

```
                  PPO/GAE bootstrap-stop          KV cache reset
                  ─────────────────────           ──────────────
trial boundary    YES  (truncations[t]=1)         no  ← preserves context
episode boundary  YES  (terminals[t]=1)           YES (fresh i.i.d. start)
```

## Cache reset gate (`pufferl.py`)

```python
done_mask = d                # was: d + t  (gated on terminals only)
self.transformer_context[done_mask.bool()] = 0
```

If we used `d + t`, every trial boundary would wipe the cache — exactly
the opposite of what we want. Trial mode breaks without this fix.

## Option D — idle-after-max_trials

Naïve trial mode would, after the agent completes `max_trials_per_episode`
trials, immediately reset the env and start a new episode. With Python's
typical rollout of one map per `resample_frequency` ticks, this leads to
**many short episodes on the same map** — the agent overfits to a tiny
subset of maps within a single Python cycle, and gradient updates see the
same map's gradients repeatedly.

Option D fixes this by **idling the agent after its episode ends**:

```c
if (e->trial_count >= env->max_trials_per_episode) {
    env->terminals[i] = 1;
    add_log_one_agent(env, i);
    e->removed       = 1;
    e->x = e->y      = INVALID_POSITION;  // off-grid
    e->vx = e->vy    = 0.0f;
    // do NOT call c_reset
}
```

The agent is invisible to subsequent `c_step`s (the top-of-loop
`if (e->removed) continue;` gates it out). It stays idle until Python's
`_reinit_envs_with_new_maps` fires at the next `resample_frequency`
boundary — that's when the env loads a fresh map and `c_reset` resets
`removed = 0`.

**Net effect**: 1 episode per resample window, exactly one fresh map per
episode. Map diversity restored.

## Auto-link of trial parameters

The user's mental model under `goal_behavior=3` is: "I'm running $k$
trials, each of length $L$." In `pufferlib/ocean/drive/adaptive.py`:

| Parameter | Auto-linked to | Override behavior |
|---|---|---|
| `max_trials_per_episode` | `k_scenarios` | Pass any value $\neq 2$ (the INI default) to disable the link |
| `per_trial_timeout` | `scenario_length` | Pass any value $> 0$ to override |
| `resample_frequency` | $k \times L$ | Pass any value $> 0$ to override |

So `--env.k-scenarios 4 --env.goal-behavior 3 --env.scenario-length 201`
gives 4 trials of 201 ticks each, episode budget = 804 ticks,
resample at tick 804.

**Gotcha.** The auto-link runs inside `AdaptiveDrivingAgent.__init__`. It
mutates `kwargs` and the resulting env attributes, **not the outer `args`
dict** that downstream code (evaluator, render) might read directly. Read
auto-linked values from `puffer_env.driver_env.<attr>`, not from
`args["env"]`. See the M7-fix evaluator commit.

## End-to-end signal flow

```
 ┌────────────────────┐
 │     C (drive.h)    │  trial_count++,  trial_ended_this_step[i] = 1
 │   c_step trial loop│  ── if last trial: terminals[i] = 1, removed = 1
 └─────────┬──────────┘
           │  zero-copy NumPy view of trial_ended_this_step (1D u8)
           ▼
 ┌────────────────────┐
 │  Python (drive.py) │  truncations[:] = 0     # top of step
 │   step()           │  vec_step(c_envs)
 │                    │  truncations[trial_ended_this_step] = 1   # mirror, gb=3 only
 │                    │  terminals already set by C if episode end
 └─────────┬──────────┘
           │  PufferLib SHM (np.bool views)
           ▼
 ┌────────────────────┐
 │    pufferl.py      │  rollout buffers store BOTH d_t and t_t
 │  rollout + GAE     │  done_mask = d            # cache-reset gate
 │                    │  bootstrap = (d + t).clamp(1)
 │                    │  δ_t = r_t + γ(1-bootstrap_t) V_{t+1} - V_t
 │                    │  Â_t = δ_t + γλ(1-bootstrap_t) Â_{t+1}
 └─────────┬──────────┘
           ▼
       PPO update
```

## Score semantic

Standard non-trial modes set `score = 1` if the agent reaches goal "well
enough" in a single scenario (frac of goals reached above a threshold,
and no collisions). Under trial mode each episode has `max_trials`
attempts, so:

* `goals_reached_this_episode` ∈ {0, 1, …, max_trials} (one increment per
  successful trial; gated by `current_goal_reached` to prevent
  over-counting within a trial)
* `frac = goals_reached_this_episode / max_trials_per_episode`
* threshold $\tau$ ladder by $k$:
  * $k = 2$: $\tau = 0.5$  (both trials must succeed for $\text{frac} > 0.5$)
  * $k \in \{3, 4\}$: $\tau = 0.8$
  * $k \geq 5$: $\tau = 0.9$
* `score = 1` iff `frac > τ AND !collided_in_episode`

## Render gates

`respawn_agent()` is shared between `goal_behavior=0` (RESPAWN, with
intentional ghost-fade post-respawn) and the trial-mode mid-episode
respawn (no ghost — agent should be fully visible immediately for
trial 2..K). The function sets `respawn_timestep = env->timestep`, and
**seven** downstream gates use `respawn_timestep != -1` as a "ghosted"
marker:

| Location | Effect when active |
|---|---|
| `drive.h:1327` | Skip self-side collision check |
| `drive.h:1342` | Skip other-as-target collision check |
| `drive.h:2409` | Force `obs[6] = 1` (post-respawn flag) |
| `drive.h:2455` | Other-car obs (self ghosted) zeroed |
| `drive.h:2457` | Other-car obs (other ghosted) zeroed |
| `drive.h:3482` | Skip 3D mesh draw (visible symptom) |
| `drive.h:3688` | Skip WOSAC track-index overlay |

In trial mode, after `respawn_agent` in the mid-episode branch we
**must** clear the flag immediately:

```c
respawn_agent(env, agent_idx);
e->respawn_timestep = -1;       // GOAL_TRIAL is NOT a ghost-fade mode
e->trial_start_timestep = env->timestep;
```

Pre-fix symptom: trial 1 rendered correctly, trials 2..K appeared empty.

## Per-trial metrics

`add_log_one_agent` (in C) is called when an agent's episode ends. It
aggregates that single agent's metrics into `env->log` (the vec_log
sink), then resets all per-entity state the agent's next episode would
otherwise inherit (respawn_timestep, current_goal_reached, the
`metrics_array` slots, etc.).

New per-trial log fields, in addition to the standard episode metrics:

| Field | Meaning |
|---|---|
| `n_trials_completed` | Trials finished this episode (always equals `max_trials` for ego under Option D) |
| `n_trials_goal_reached` | Of those, how many reached goal |
| `n_trials_timed_out` | Of those, how many timed out |
| `trial_total_length` | Sum of trial lengths (ticks) |
| `trial_mean_length` | `trial_total_length / n_trials_completed` (computed in `add_log`) |
| `trial_goal_reach_rate` | `n_trials_goal_reached / n_trials_completed` |

These are populated **only** under `goal_behavior=3`. The standard
metrics (score, collision_rate, episode_length, …) still populate via
the same `add_log_one_agent` path.

The evaluator (`HumanReplayEvaluator`) computes additional per-trial
breakdowns from its own success array:

| Field | Definition |
|---|---|
| `trial_K_score` | $\Pr$(reached in trial $K$) over the eval rollouts |
| `ada_delta_trial_K_minus_0` | `trial_K_score - trial_0_score` (the in-context adaptation signal) |

For $K$ = `max_trials_per_episode` = 4 (auto-link from `k_scenarios=4`),
that's `trial_0_score`, …, `trial_3_score` and `ada_delta_trial_{1,2,3}_minus_0`.

## Test coverage

| File | What it covers |
|---|---|
| `test_goal_trial.py` | Trial timer fires; episode boundary fires at `trial_count == max_trials`; non-regression for gb∈{0,1,2} |
| `test_trial_ended_buffer.py` | `trial_ended_this_step` Python ↔ C buffer plumbing |
| `test_trial_log_fields.py` | Per-trial Log fields populate |
| `test_trial_standard_metrics.py` | Standard episode metrics still populate via `add_log_one_agent` |
| `test_trial_per_scenario_gate.py` | Per-scenario logic gated off under gb=3 |
| `test_trial_score_semantics.py` | Score uses `max_trials_per_episode` denominator |
| `test_trial_overcounting_fix.py` | `current_goal_reached` gates `goals_reached_this_episode` increments |
| `test_gae_trial_boundary.py` | GAE bootstrap-stop fires on truncations |
| `test_gae_decoupling_integration.py` | End-to-end `trial_ended_this_step → truncations` mirror |
| `test_adaptive_trial_link.py` | Auto-link of `max_trials_per_episode` and `per_trial_timeout` |
| `test_rollout_trial_mode.py` | Rollout `max_steps` / break / info under trial mode |
| `test_evaluator_trial_mode.py` | `HumanReplayEvaluator` emits `trial_K_score` + auto-link case |
| `test_pe_train_eval_consistency.py` | Transformer PE indexing matches between train and eval |
| `test_pos_within_episode.py` | `compute_pos_within_episode` correctness |

All 53 tests pass on `mohit/trial-episode-redesign` HEAD.

## Quick reference

```
goal_behavior   = 3                          # the toggle
k_scenarios     = 4                          # the number of trials, by auto-link
scenario_length = 201                        # nuplan trajectories are 201 ticks
                                             # → per_trial_timeout = 201, by auto-link
                                             # → episode budget = 804 ticks
                                             # → resample_frequency = 804, by auto-link
```

| Knob | Type | Default | Notes |
|---|---|---|---|
| `--env.goal-behavior` | int | 0 | 3 = trial mode |
| `--env.max-trials-per-episode` | int | 2 (INI) | Auto-linked to `k_scenarios` if left at default |
| `--env.per-trial-timeout` | int | 0 (= `scenario_length`) | Auto-linked to `scenario_length` if 0 |
| `--env.k-scenarios` | int | 1 | Driver of auto-link |
| `--env.scenario-length` | int | 91 | Driver of auto-link |
