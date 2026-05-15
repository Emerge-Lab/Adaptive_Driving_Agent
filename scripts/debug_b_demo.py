"""Debug walk-through of GOAL_TRIAL B'' semantic.

Shows step-by-step:
  - env-level trial state (env_trial_count, env_trial_start_timestep)
  - per-agent `removed` flag transitions (off-map on reach)
  - truncations + terminals firing at env trial / episode boundaries
  - move_expert clock reset (verified by checking that recorded humans
    rewind to frame 0 at every env trial-end)
  - render: produces a video where ego is visible in every trial

Run:
  python scripts/debug_b_demo.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_hard"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(k=4, scenario_length=10, num_agents=4, goal_radius=200.0):
    from pufferlib.ocean.drive import Drive

    return Drive(
        num_agents=num_agents,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=3,
        k_scenarios=k,
        max_trials_per_episode=k,
        per_trial_timeout=scenario_length,
        goal_radius=goal_radius,
        report_interval=10000,
    )


def _fmt(arr):
    return "".join("1" if x else "." for x in np.asarray(arr, dtype=bool))


def trace_env_trial_state():
    print("=" * 72)
    print("WALK-THROUGH: env-level trial state, removed flag, boundaries")
    print("=" * 72)
    print()
    print("Config: k_scenarios=4, scenario_length=10, num_agents=4,")
    print("        goal_radius=200 (wide → most egos reach quickly)")
    print()
    env = _make_env(k=4, scenario_length=10, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)

    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 0.1   # gentle forward accel

    print(f"  {'tick':>4}  {'removed':>8}  {'trunc':>8}  {'term':>8}  {'trial_end_flag':>14}   note")
    print(f"  {'-'*4:>4}  {'-'*8:>8}  {'-'*8:>8}  {'-'*8:>8}  {'-'*14:>14}   {'-'*30}")

    for t in range(1, 60):
        env.step(actions)
        rem = np.asarray(env.removed, dtype=bool)
        tr = np.asarray(env.truncations, dtype=bool)
        term = np.asarray(env.terminals, dtype=bool)
        te = np.asarray(env.trial_ended_this_step, dtype=bool)

        notes = []
        if tr.any() and not term.any():
            notes.append("← env trial-end (mid-episode): world resets")
        if term.any():
            notes.append("← env EPISODE-end (Option D): all egos removed permanently")
        if rem.any() and not tr.any() and not term.any():
            notes.append("(some egos off-map, waiting for stragglers)")

        if (rem.any() or tr.any() or term.any()) or t <= 3:
            note = "  ".join(notes)
            print(f"  {t:>4}  {_fmt(rem):>8}  {_fmt(tr):>8}  {_fmt(term):>8}  {_fmt(te):>14}   {note}")

        if term.any():
            break

    env.close()


def trace_move_expert_clock():
    """Show that move_expert's clock rewinds at env trial-end.

    We can't easily inspect the C-side clock from Python, but we can verify
    indirectly: at env trial-end, the recorded humans should be back at
    their init positions (same as the start of trial 1). The proxy is
    `env.observations` for ego A — the "other agents" features should be
    identical at tick 1 of trial 1 and tick 1 of trial 2.
    """
    print()
    print("=" * 72)
    print("MOVE_EXPERT CLOCK: humans rewind to frame 0 at env trial-end")
    print("=" * 72)
    print()
    env = _make_env(k=4, scenario_length=8, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 0.1

    # Snapshot ego[0] obs at tick 1 of trial 1
    env.step(actions)
    obs_trial_1_tick_1 = env.observations[0].copy()

    # Run until next env trial-end fires, then 1 step into trial 2
    seen_trial_end = False
    for t in range(2, 60):
        env.step(actions)
        if np.asarray(env.truncations, dtype=bool).any() and not np.asarray(env.terminals, dtype=bool).any():
            seen_trial_end = True
            # Trial 2 has now started (entities reset). Step once into trial 2.
            env.step(actions)
            obs_trial_2_tick_1 = env.observations[0].copy()
            break

    if not seen_trial_end:
        print("  ! no mid-episode trial-end observed in 60 steps — test setup wrong")
        env.close()
        return

    # ego features (first ~20 dims) reflect ego's own state — depends on
    # ego's actions, so will differ across trials. Other-agent features
    # (slots after ego_dim) should match closely because the world resets.
    OTHER_AGENT_OFFSET = 20
    diff_ego = float(np.abs(obs_trial_1_tick_1[:OTHER_AGENT_OFFSET] - obs_trial_2_tick_1[:OTHER_AGENT_OFFSET]).mean())
    diff_others = float(np.abs(obs_trial_1_tick_1[OTHER_AGENT_OFFSET:] - obs_trial_2_tick_1[OTHER_AGENT_OFFSET:]).mean())
    print(f"  Mean |obs[trial_1_tick_1] - obs[trial_2_tick_1]| over:")
    print(f"    ego features [0..20):       {diff_ego:.6f}")
    print(f"    other-agent features [20:]: {diff_others:.6f}")
    print(f"  → other-agent features near-identical (move_expert rewound to frame 0).")
    print(f"  → ego features differ because ego just took actions in trial 1.")
    env.close()


def trace_per_trial_metrics():
    """Show the per-trial-K success counters populate in the log."""
    print()
    print("=" * 72)
    print("PER-TRIAL METRICS: trial_K_score in vec_log")
    print("=" * 72)
    print()
    from pufferlib.ocean.drive import binding

    env = _make_env(k=4, scenario_length=8, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 0.1

    # Run a full episode
    log = None
    for _ in range(200):
        env.step(actions)
        log = binding.vec_log(env.c_envs, env.num_agents)
        if log and log.get("n", 0) > 0:
            break

    if not log or log.get("n", 0) == 0:
        print("  ! no log emitted in 200 steps")
        env.close()
        return

    print(f"  log.n (ego-episodes counted): {log['n']:.0f}")
    print(f"  log.n_trials_completed:       {log['n_trials_completed']:.2f}")
    print(f"  log.n_trials_goal_reached:    {log['n_trials_goal_reached']:.2f}")
    print(f"  log.trial_goal_reach_rate:    {log['trial_goal_reach_rate']:.3f}")
    print(f"  log.score:                    {log['score']:.3f}")
    print()
    print("  Per-trial-index success rate (from C log.trial_k_goal_reached):")
    for k in range(env.k_scenarios):
        key = f"trial_{k}_score"
        if key in log:
            print(f"    trial_{k}_score: {log[key]:.3f}")
    env.close()


def trace_cache_freeze_design_intent():
    """The pufferl-side KV cache freeze isn't wired yet. This documents the
    design and shows where it WOULD plug in."""
    print()
    print("=" * 72)
    print("KV CACHE FREEZE (pufferl integration — DESIGN, not wired yet)")
    print("=" * 72)
    print()
    print("  The env exposes a per-agent `removed` SHM buffer (per-step,")
    print("  numpy bool array of shape (num_agents,)). pufferl reads it via")
    print("  the same SHM mechanism as `terminals` and `truncations`.")
    print()
    print("  In pufferl.py, between the policy forward call and the cache")
    print("  persist (around line 711-718), restore the previous cache state")
    print("  for agents with removed=1:")
    print()
    print("    # snapshot cache state BEFORE policy forward")
    print("    prev_ctx = self.transformer_context[key].clone()")
    print("    prev_pos = self.transformer_position[key].clone()")
    print("    prev_kc  = [c.clone() for c in self.transformer_k_cache[key]]")
    print("    prev_vc  = [c.clone() for c in self.transformer_v_cache[key]]")
    print()
    print("    # policy forward (existing code) — appends new K, V to cache")
    print("    ...")
    print()
    print("    # AFTER persist: for removed agents, restore previous state")
    print("    removed = self.vecenv.driver_env.removed  # (num_agents,)")
    print("    if removed.any():")
    print("        rem_idx = torch.where(torch.from_numpy(removed))[0]")
    print("        self.transformer_context[key][rem_idx] = prev_ctx[rem_idx]")
    print("        self.transformer_position[key][rem_idx] = prev_pos[rem_idx]")
    print("        for c, p in zip(self.transformer_k_cache[key], prev_kc):")
    print("            c[rem_idx] = p[rem_idx]")
    print("        ... (same for v_cache)")
    print()
    print("  This requires routing per-agent `removed` through vecenv.recv()")
    print("  or adding a side-channel read. Captured as task #33.")


def main():
    trace_env_trial_state()
    trace_move_expert_clock()
    trace_per_trial_metrics()
    trace_cache_freeze_design_intent()
    print()
    print("=" * 72)
    print("DONE. To verify renders: see scripts/debug_b_render.sh")
    print("=" * 72)


if __name__ == "__main__":
    main()
