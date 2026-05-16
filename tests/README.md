# Tests

## Trial-mode contract tests (GOAL_TRIAL, gb=3)

| File | Guards |
|---|---|
| `test_goal_trial.py` | gb=3 timer + episode boundaries; non-regression for gb∈{0,1,2} |
| `test_env_level_trial.py` | trial timing + bit-identical re-start across trials |
| `test_cache_freeze.py` | KV slot persists (no overwrite) when `removed=1` |
| `test_transformer_kv_cache.py` | train/eval attention-mask equivalence |
| `test_gae_trial_boundary.py` | bootstrap-stop respects `removed` |
| `test_trial_log_fields.py` | `n_trials_*` + `trial_*_rate` populate in logs |
| `test_drive_train.py` | base drive training still runs |

## Running

Drive tests share raylib global state, so run each file in its own pytest
process:

```bash
for t in tests/test_*.py; do
  python -m pytest "$t" -q
done
```

## Reference

Design spec: [`docs/src/trial_mode.md`](../docs/src/trial_mode.md).
