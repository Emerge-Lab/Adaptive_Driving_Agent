# Tests

## Contract vs regression

Trial-mode tests come in two categories:

* **Contract tests** assert design invariants — they should pass on **any
  correct implementation** of trial mode. Read them to understand the
  semantics; write them when adding new functionality.

* **Regression tests** lock in specific bug fixes — they exist only because
  a bug shipped and we don't want it back. Read them to learn about past
  pitfalls; don't write new ones unless you're fixing a real bug.

Going forward, write contract tests **before** the implementation. The
regression-heavy state of the suite below reflects honest practice during
the initial build-out; we're correcting course.

## Trial-mode test files

| File | Category | Asserts |
|---|---|---|
| `test_goal_trial.py` | contract | gb=3 timer/episode boundaries; non-regression for gb∈{0,1,2} |
| `test_trial_ended_buffer.py` | contract | C↔Python `trial_ended_this_step` buffer plumbing |
| `test_trial_log_fields.py` | contract | `n_trials_*` and `trial_*_rate` log fields populate |
| `test_trial_per_scenario_gate.py` | contract | per-scenario block gated off under gb=3 |
| `test_adaptive_trial_link.py` | contract | `k_scenarios` / `scenario_length` are canonical; overrides ignored |
| `test_gae_trial_boundary.py` | contract | GAE bootstrap-stop = terminals ∨ truncations |
| `test_render_contract.py` | contract | ego visible across all trials (`respawn_timestep != -1` gate clears) |
| `test_truncations_ownership.py` | contract | C is the only writer of `truncations`/`trial_ended_this_step` under gb=3 |
| `test_ada_delta_train_logging.py` | contract | `trial_K_score` + `ada_delta_trial_K_minus_0` in training logs |
| `test_evaluator_trial_mode.py` | contract | `HumanReplayEvaluator` emits per-trial breakdown under gb=3 |
| `test_pe_train_eval_consistency.py` | contract | Transformer PE indexing matches train/eval |
| `test_pos_within_episode.py` | contract | `compute_pos_within_episode` correctness |
| `test_trial_overcounting_fix.py` | regression | `current_goal_reached` gates `goals_reached_this_episode` |
| `test_trial_score_semantics.py` | regression (partial) | score uses `max_trials` denominator under gb=3 |
| `test_trial_standard_metrics.py` | regression | standard metrics still populate via `add_log_one_agent` |
| `test_rollout_trial_mode.py` | regression | rollout `max_steps` / break / info match trial mode |
| `test_gae_decoupling_integration.py` | regression | end-to-end `trial_ended_this_step → truncations` (now C-side, still valid) |

## Running

Some Drive tests segfault when run in the same pytest process because
raylib's global state doesn't tear down cleanly across multiple `Drive(...)`
instantiations. Workaround: run each test file in its own pytest
invocation:

```bash
for t in tests/test_*.py; do
  python -m pytest "$t" -q
done
```

A real fix would be a module-scoped fixture with explicit raylib cleanup.
Captured as future work in `notes/trial_episode_design.md`.

## Reference

The full trial-mode design spec lives at
[`docs/src/trial_mode.md`](../docs/src/trial_mode.md). Tests should be
readable against it.
