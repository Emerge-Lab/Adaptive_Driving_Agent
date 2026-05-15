"""Per-step text trace of a B'' trial-mode rollout.

Loads the same checkpoint render.py uses and runs a single rollout, printing
per step:
  - tick, env_trial_count (inferred from truncations)
  - terminals, truncations, trial_ended_this_step (count of agents firing)
  - removed (count of egos off-map mid-trial)
  - KV cache write position (transformer_position scalar)
  - mean partner-obs energy (proxy for "are humans/co-players visible")
  - per-event highlights: GOAL-REACH, TRIAL-END, EPISODE-END

Skip render frames — this is a text-only inspection tool.

Run:
  python scripts/trace_b_render.py
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="/tmp/ADA-work/experiments/puffer_adaptive_drive_rrvyie58/model_puffer_adaptive_drive_000400.pt",
    )
    ap.add_argument("--map-dir", default="resources/drive/binaries/nuplan_hard")
    ap.add_argument("--num-maps", type=int, default=50)
    ap.add_argument("--num-agents", type=int, default=64)
    ap.add_argument("--num-ego-agents", type=int, default=32)
    ap.add_argument("--k-scenarios", type=int, default=4)
    ap.add_argument("--scenario-length", type=int, default=201)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=804)
    args = ap.parse_args()

    from pufferlib.ocean.drive.adaptive import AdaptiveDrivingAgent

    env = AdaptiveDrivingAgent(
        num_agents=args.num_agents,
        num_ego_agents=args.num_ego_agents,
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        scenario_length=args.scenario_length,
        k_scenarios=args.k_scenarios,
        goal_behavior=3,
        dynamics_model="classic",
        co_player_enabled=False,
    )
    env.reset(seed=args.seed)

    # Try to load the policy; fall back to zero actions if state-dict mismatches
    policy = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.model_path, map_location=device)
        from pufferlib.models import TransformerWrapper  # noqa: F401
        # Best-effort: skip policy loading; the trace works with zero actions
        # to focus on env state evolution.
        print(f"[trace] (policy loading skipped — using zero actions to focus on env semantics)\n")
    except Exception as e:
        print(f"[trace] policy load failed ({e!r}); using zero actions\n")

    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0  # full accel — like a trained-ish policy

    print(
        f"Config: k_scenarios={args.k_scenarios}, scenario_length={args.scenario_length}, "
        f"num_egos={args.num_ego_agents}, goal_behavior=3 (B'')"
    )
    print(f"Episode budget: {args.k_scenarios * args.scenario_length} ticks")
    print()
    print(
        f"{'tick':>5}  {'trial':>5}  {'rem':>4}  {'TE':>3}  {'trunc':>5}  {'term':>4}  "
        f"{'cache_pos':>9}  {'partner_obs':>11}   event"
    )
    print(
        f"{'-'*5:>5}  {'-'*5:>5}  {'-'*4:>4}  {'-'*3:>3}  {'-'*5:>5}  {'-'*4:>4}  "
        f"{'-'*9:>9}  {'-'*11:>11}   {'-'*30}"
    )

    cache_pos = 0  # simulate transformer cache write position
    trial_idx = 1  # 1-based for the overlay
    rem_prev = np.zeros(env.num_agents, dtype=bool)

    for t in range(1, args.max_steps + 1):
        env.step(actions)
        rem = np.asarray(env.removed, dtype=bool)
        te = np.asarray(env.trial_ended_this_step, dtype=bool)
        tr = np.asarray(env.truncations, dtype=bool)
        term = np.asarray(env.terminals, dtype=bool)
        partner_obs = float(np.abs(env.observations[0, 20:]).mean())

        # Cache position semantics: each step advances cache by 1, EXCEPT
        # for agents with removed=1 (frozen cache, task #33), and resets to
        # 0 on terminals. We track the ego-0 perspective scalar.
        if term[0] if term.shape[0] > 0 else False:
            cache_pos_after = 0
            cache_event = "← KV cache reset (terminals)"
        elif rem[0] if rem.shape[0] > 0 else False:
            cache_pos_after = cache_pos  # frozen (would be, with task #33 wired)
            cache_event = "  cache frozen (ego off-map)"
        else:
            cache_pos_after = cache_pos + 1
            cache_event = ""

        # Detect goal-reach events: removed transition 0 → 1 (new reachers this step)
        new_reached = (~rem_prev) & rem
        events = []
        if new_reached.any():
            events.append(f"GOAL-REACH agents={list(np.where(new_reached)[0])[:6]}")
        if tr.any() and not term.any():
            events.append(f"TRIAL-END (→ trial {trial_idx + 1})")
            trial_idx += 1
        if term.any():
            events.append("EPISODE-END (Option D)")
        if cache_event and not events:
            events.append(cache_event.strip())

        # Print every step for the first 30, then every trial-boundary or event
        is_interesting = t <= 5 or new_reached.any() or tr.any() or term.any() or t % 30 == 0
        if is_interesting:
            event_str = "  ".join(events)
            print(
                f"{t:>5}  {trial_idx:>5}  {int(rem.sum()):>4}  {int(te.sum()):>3}  "
                f"{int(tr.sum()):>5}  {int(term.sum()):>4}  "
                f"{cache_pos_after:>9}  {partner_obs:>11.3f}   {event_str}"
            )

        cache_pos = cache_pos_after
        rem_prev = rem.copy()
        if term.any():
            break

    print()
    print("Legend:")
    print("  rem    = count of egos currently off-map (post-reach, awaiting trial-end)")
    print("  TE     = count of agents with trial_ended_this_step set this tick")
    print("  trunc  = count of agents with truncations set this tick")
    print("  term   = count of agents with terminals set this tick")
    print("  cache_pos = simulated transformer KV-cache write position for ego 0")
    print("              (advances each step, resets to 0 on terminals)")
    print()
    print("Expected B'' invariants:")
    print("  - GOAL-REACH events: rem count increases, but trunc/term stay 0 until trial-end")
    print("  - TRIAL-END events: trunc>0 + term=0; all rem flags reset to 0 same step")
    print("  - EPISODE-END: term>0; rem stays high (Option D off-map until c_reset)")
    print("  - cache_pos: advances through trials (cache spans the episode); resets at term")
    print()
    print("Once task #33 is wired (pufferl cache freeze for removed agents), cache_pos")
    print("won't advance during 'cache frozen' steps — the cache will be exactly equal")
    print("at trial K tick N and trial 1 tick N + |cumulative cache from trials 1..K-1|.")

    env.close()


if __name__ == "__main__":
    main()
