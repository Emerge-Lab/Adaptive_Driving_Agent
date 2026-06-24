# `nuplan_hard` map split

## Why this exists

Default eval (`--eval.map-dir resources/drive/binaries/nuplan_201`, 50 rollouts) shows
`ada_delta_score ≈ ±0.005` across all configs. Looks like the policy doesn't
adapt across scenarios.

Diagnosis (from `score_maps_interaction.py` over the 5401 maps in nuplan_201):

| Property | Count |
|----------|------:|
| Total maps | 5401 |
| Maps where SDC has *any* moving vehicle within 5m at any timestep | 2497 (46%) |
| Maps where SDC has zero such interaction-steps | **2904 (54%)** |

**Over half of all maps have zero ego-other interaction.** On those maps the
ego just drives straight and `ada_delta` collapses to noise. The signal exists,
but the average is dominated by maps where adaptation is irrelevant.

## What `nuplan_hard` is

A symlink-only directory at `resources/drive/binaries/nuplan_hard/` containing
the **top 10% of nuplan_201 maps by SDC-vehicle interaction density** (540 maps,
sequentially renumbered as `map_001.bin` through `map_540.bin`).

Threshold: maps with `sdc_interaction_steps ≥ 52`. Mean in the hard set: 79
steps (out of 91-201 timesteps per scenario).

Each `map_NNN.bin` is a symlink pointing back to the original map in
`nuplan_201/`. The mapping (new bin id ↔ original bin id ↔ scenario_id) is
recorded in `_manifest.csv` for traceability.

## How to recreate

The split is reproducible. To regenerate from the underlying JSON
trajectories in `data/nuplan_gpudrive/nuplan/`:

```bash
# 1. Score every map by SDC interaction-steps (~3 min, multi-process)
python scripts/score_maps_interaction.py \
    --data-dir data/nuplan_gpudrive/nuplan \
    --out /tmp/nuplan_201_hardness_scores.csv

# 2. Build the symlink directory from the top-10% slice
python scripts/build_nuplan_hard.py \
    --scores /tmp/nuplan_201_hardness_scores.csv \
    --source-dir resources/drive/binaries/nuplan_201 \
    --out-dir   resources/drive/binaries/nuplan_hard \
    --metric    sdc_interaction_steps \
    --top-pct   10
```

Different threshold or metric? Pass other values to step 2 (`--top-pct 25`,
`--metric interaction_events`, etc.). The scoring step doesn't have to be
re-run — its CSV holds both metrics for every map.

## Hardness definition (operational)

For each scenario in the JSON dataset:

1. Filter to vehicle-type agents only (excludes pedestrians, cyclists).
2. Identify the SDC (the `is_sdc=True` agent — this is the slot the ego
   policy occupies during human-replay eval).
3. For each timestep `t`:
   - Skip if SDC is not valid at `t`, or is parked (speed < 0.5 m/s).
   - Find any other vehicle that is valid AND moving (speed > 0.5 m/s)
     AND within Euclidean distance 5 m of the SDC.
   - If at least one such vehicle exists, count this timestep as an
     SDC interaction-step.
4. `sdc_interaction_steps` = total such timesteps for the scenario.

This is purely a property of the map + recorded human trajectories — no
trained policy is involved. A map is "hard" iff the SDC's logged trajectory
brings it close to another moving vehicle, often. That's exactly when our
trained ego (which replaces the SDC at eval time) faces real interactions
to adapt to.

## Eval against `nuplan_hard`

Stock command (replace checkpoint and GPU):

```bash
xvfb-run -a puffer eval puffer_adaptive_drive \
  --load-model-path  experiments/puffer_adaptive_drive_<runid>/model_..._N.pt \
  --policy-architecture Transformer --rnn-name Transformer \
  --train.horizon 402 \
  --vec.num-workers 1 --vec.num-envs 1 \
  --env.map-dir              resources/drive/binaries/nuplan_hard \
  --env.num-maps             540 \
  --env.scenario-length      201 \
  --env.k-scenarios          2 \
  --env.conditioning.type    none \
  --env.co-player-enabled    1 \
  --env.co-player-policy.policy-path  experiments/puffer_drive_2e029h15.pt \
  --env.co-player-policy.architecture Transformer \
  --env.co-player-policy.transformer.horizon 201 \
  --env.co-player-policy.conditioning.type all \
  --env.co-player-policy.conditioning.collision-weight-lb -2 \
  --env.co-player-policy.conditioning.collision-weight-ub 0 \
  --env.co-player-policy.conditioning.offroad-weight-lb -2 \
  --env.co-player-policy.conditioning.offroad-weight-ub 0 \
  --env.co-player-policy.conditioning.entropy-weight-lb 0 \
  --env.co-player-policy.conditioning.entropy-weight-ub 0.10 \
  --env.co-player-policy.conditioning.discount-weight-lb 0.4 \
  --env.co-player-policy.conditioning.discount-weight-ub 1 \
  --env.map-rand-per-scenario False \
  --eval.map-dir                  resources/drive/binaries/nuplan_hard \
  --eval.num-maps                 540 \
  --eval.human-replay-eval        True \
  --eval.human-replay-num-rollouts 200 \
  --eval.human-replay-num-maps    540 \
  --eval.human-replay-num-agents  540
```

Note `--eval.human-replay-num-{rollouts,maps,agents}` are required:
without them, eval defaults to 100 maps / 100 rollouts which both
under-samples and may not all hit the hard split.

## Result on baseline checkpoint

On `puffer_adaptive_drive_4lm6kkh7/model_..._40.pt` (γ=0.995, partner 2e029h15)
with 200 rollouts:

| Metric | Full set (50 rollouts) | `nuplan_hard` (200 rollouts) |
|--------|-----------------------:|-----------------------------:|
| `ada_delta_score` | ±0.005 | **+0.222 ± 0.18** |
| `ada_delta_collision_rate` | ~0 | -0.045 |
| `ada_delta_episode_return` | ~0 | +0.449 |
| `ada_delta_dnf_rate` | ~0 | -0.170 |
| `scenario_0_score` | — | 0.712 |
| `scenario_1_score` | — | **0.935** |

The model **does** adapt (s_1 score 22% higher than s_0; collisions cut roughly
in half). The full-set average diluted the signal by 40-200×.

## Caveats / things to keep in mind

- "Hard" here is a property of the SDC's *recorded* trajectory in the
  human-replay data. The trained ego is free to take a different path; a
  defensive policy may avoid the close-proximity moments that scored the map
  hard in the first place. The split is a *prior* over which maps are likely
  to involve interaction, not a guarantee.
- Threshold (5 m, 0.5 m/s, top 10%) is somewhat arbitrary. Re-running with
  different thresholds will produce different splits — `_manifest.csv` records
  the actual threshold used.
- Eval channel is human-replay (other vehicles = recorded humans, no synthetic
  partner). The hard-set definition matches this channel — for synthetic-
  partner eval channels you'd want a different definition (e.g.
  partner-sensitivity-based).
