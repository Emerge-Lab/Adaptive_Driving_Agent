# Adaptive-Driving — Final Training Runs Manifest

Generated 2026-06-21. These are the **final Step-2 ego runs** for the design-B
adaptation experiment: 45 runs = **3 co-players × 5 k × 3 seeds**.
(The 4th co-player column, `2e029h15` @ entropy_ub=0.10, is the 15 pre-existing
runs from the earlier main sweep — see "Full 60-cell grid" note at the bottom.)

---

## Experiment design (B, 2-step)

**Step 1 — conditioned co-players.** A single conditioned co-player net is trained
per entropy upper-bound. The co-player is conditioned on (collision, offroad,
entropy, discount) weights; for Step 2 we pin all conditioning except `entropy_ub`,
which *is* the partner identity. Higher `entropy_ub` = more stochastic / less
predictable partner.

| partner id | entropy_ub | Step-1 checkpoint |
|---|---|---|
| `miku2puk`  | 0.05 | `experiments/puffer_drive_miku2puk.pt` |
| `2e029h15`  | 0.10 | `experiments/puffer_drive_2e029h15.pt` (used by the prior sweep) |
| `m2ygolog`  | 0.20 | `experiments/puffer_drive_m2ygolog.pt` |
| `6rauydj2`  | 0.50 | `experiments/puffer_drive_6rauydj2.pt` |

**Step 2 — ego vs a SINGLE conditioned co-player.** The ego policy is trained
against one frozen co-player net, varying the trial length k ∈ {2,3,4,5,6} and
seed ∈ {42,43,44}. Eval is on **human logs** (`nuplan_hard`, 540 maps,
human-replay), which feeds the per-map adapted/flat/regressed analysis.

---

## Step-2 training recipe (shared)

Identical across all 45 runs except where noted (k5/k6 use a memory-safe vec config).

- entrypoint: `puffer train puffer_adaptive_drive` (Transformer policy + Transformer rnn)
- wandb project: `adaptive_aligned_v2`; tag: `ada_k${K}_gb3_legacy_eval_fix` (per-k; partner+entropy_ub live in config)
- total timesteps: **3e9** (3B); checkpoint-interval 10; eval-interval 10
- optimizer: lr `3e-3`, ent-coef `0.005`, gamma `0.995`, cosine LR over the full run
- env: `nuplan_201` map-dir, num-maps 4999, scenario-length 201, **goal-behavior 3**,
  horizon = k×201, lane-align reward 0.05, collision/offroad ego penalty −0.5
- co-player: `external-co-player-actions True`, conditioning.type all, entropy-weight-ub = partner's entropy_ub,
  collision/offroad weight-lb −2/ub 0, discount-lb 0.4/ub 1
- eval: `nuplan_hard`, human-replay, 10 rollouts; eval agents/maps 540 (k2–k4) or 270 (k5/k6, memsafe)
- env flag: `PUFFER_TRANSFORMER_LEGACY_EVAL=1` (full-context eval forward; required for gb=3)

**Submit scripts**
- k2/k3/k4: `scripts/adaptive/cluster_coplayer_grid_k234.sh` (job 10993787, nw=nv=32, batch-size 32, mem 256GB)
- k5/k6:    `scripts/adaptive/cluster_coplayer_grid_k56.sh`  (job 10993788, nw=nv=8, batch-size 8, mem 600GB, eval agents 270)
- resume:   `scripts/adaptive/cluster_coplayer_grid_resume.sh` (job 11287601) — continues sub-target cells from their last checkpoint

**Per-k natural 3B end (iter target).** k2≈228, k3=152, k4=114, k5=365, k6=304.
Final checkpoint sizes (bytes): k2 5,357,453 · k3 5,563,277 · k4 5,769,101 · k5 5,974,925 · k6 6,180,749.

**Resume mechanism.** `--load-model-path <model_NNNNNN.pt>` restores weights + Adam
state + `global_step` + cosine-LR position (via sibling `trainer_state.pt`) so the
loop trains only the remainder; `--load-id <wid>` keeps the same wandb run continuous.

---

## All 45 runs

Checkpoint path = `experiments/puffer_adaptive_drive_<wandb_id>/model_puffer_adaptive_drive_<iter>.pt`.
Status: **final** = clean checkpoint at iter target; **resuming** / **rerun** = job in flight,
iter shown is current max and will advance to target on completion.

### miku2puk — entropy_ub 0.05

| k | seed | wandb id | iter/target | status |
|---|---|---|---|---|
| 2 | 42 | `6opvas42` | 228/228 | final |
| 2 | 43 | `3dlsmo4v` | 228/228 | final |
| 2 | 44 | `6gzqx4gj` | 228/228 | final |
| 3 | 42 | `io7kp2cq` | 152/152 | final |
| 3 | 43 | `xr137mjs` | 152/152 | final |
| 3 | 44 | `rf4p3hy6` | 152/152 | final |
| 4 | 42 | `9gc19bcy` | 114/114 | final |
| 4 | 43 | `bhx6zxn0` | 80→114 | **resuming** (11287601_7) |
| 4 | 44 | `9qewt905` | 50→114 | **rerun** (11131115_8) |
| 5 | 42 | `246wih6m` | 365/365 | final |
| 5 | 43 | `w1u39swd` | 365/365 | final |
| 5 | 44 | `f754lim2` | 365/365 | final |
| 6 | 42 | `ijwa3y93` | 304/304 | final |
| 6 | 43 | `hk5gtm99` | 304/304 | final |
| 6 | 44 | `yl6tfvb0` | 304/304 | final |

### m2ygolog — entropy_ub 0.20

| k | seed | wandb id | iter/target | status |
|---|---|---|---|---|
| 2 | 42 | `obrwxqqy` | 228/228 | final |
| 2 | 43 | `gesyc4j9` | 228/228 | final |
| 2 | 44 | `s790xh0d` | 228/228 | final |
| 3 | 42 | `052n7brp` | 90→152 | **resuming** (11287601_12) |
| 3 | 43 | `n46bkreg` | 152/152 | final |
| 3 | 44 | `cei22yc1` | 130→152 | **resuming** (11287601_14) |
| 4 | 42 | `ftxa55g3` | 114/114 | final |
| 4 | 43 | `citbzhdc` | 10→114 | **resuming** (11287601_16) |
| 4 | 44 | `c0k9uqhc` | 114/114 | final |
| 5 | 42 | `72dilduo` | 365/365 | final |
| 5 | 43 | `06uc7cis` | 365/365 | final |
| 5 | 44 | `e35ds248` | 365/365 | final |
| 6 | 42 | `0r5j0p8y` | 304/304 | final |
| 6 | 43 | `osjfnxz0` | 304/304 | final |
| 6 | 44 | `w8sbas7o` | 304/304 | final |

### 6rauydj2 — entropy_ub 0.50

| k | seed | wandb id | iter/target | status |
|---|---|---|---|---|
| 2 | 42 | `cgq46nnk` | 228/228 | final |
| 2 | 43 | `c8yihf5o` | 220/228 (96%) | final |
| 2 | 44 | `yw4mao1d` | 228/228 | final |
| 3 | 42 | `x4cskyot` | 152/152 | final |
| 3 | 43 | `blyerjec` | 140→152 | **resuming** (11287601_22) |
| 3 | 44 | `1eqwuq6m` | 152/152 | final |
| 4 | 42 | `m4ibxlhu` | 114/114 | final |
| 4 | 43 | `yu0vk259` | 114/114 | final |
| 4 | 44 | `7s0p1b8q` | 114/114 | final |
| 5 | 42 | `hzttkj82` | 365/365 | final |
| 5 | 43 | `fkufm4ol` | 365/365 | final |
| 5 | 44 | `071oqqu5` | 365/365 | final |
| 6 | 42 | `p4ltvhdf` | 304/304 | final |
| 6 | 43 | `9ti3ocxm` | 304/304 | final |
| 6 | 44 | `ly30vz8m` | 304/304 | final |

---

## Status summary (2026-06-21)

- **39/45 final** (clean checkpoint at iter target; `c8yihf5o` at 96% counted as final).
- **5 resuming** via job 11287601: `bhx6zxn0`, `052n7brp`, `cei22yc1`, `citbzhdc`, `blyerjec`.
- **1 rerun from scratch** via job 11131115_8: `9qewt905` (original cell was scheduler-cancelled).
- All TIMEOUT-state cells that hit the 24h k234 wall but reached their iter target are
  **benign** (only the trailing post-train eval was SIGTERM'd; the final checkpoint is valid).

## Next steps

- **Task #30** — eval all 45 final checkpoints at 540×20 on human logs
  (`scripts/adaptive/eval_final_540.py` + `cluster_eval540_all.sh`). The driver needs
  (k, seed, wandb id, final iter) — read them from the tables above once the 6 in-flight
  cells reach target.
- **Task #31** — per-map adaptation analysis (F1–F4): ada_delta vs entropy_ub by k,
  classifying maps adapted / flat / regressed (`scripts/adaptive/analyze_eval540_f1f4.py`).

## Full 60-cell grid note

These 45 cover entropy_ub ∈ {0.05, 0.20, 0.50}. The **0.10 column** (`2e029h15`,
15 runs: 5 k × 3 seeds) comes from the earlier main sweep and is *not* listed here.
For the full 60-cell adaptation analysis across all four entropy levels, fold those
15 wandb ids in alongside this table.
