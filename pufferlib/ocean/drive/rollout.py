"""Shared rollout loop for Drive evaluation and rendering.

Single source of truth for the forward-sample-step-break cycle. Used by:
  - Training renders (periodic evaluation with video logging)
  - Offline batch rendering
  - Safe evaluation time rendering

This module enables rendering with ANY policy architecture (LSTM, Transformer, etc.)
by keeping policy inference in Python/PyTorch while using C bindings for environment
simulation and graphics rendering.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import torch

import pufferlib.pytorch


class RenderView(IntEnum):
    """View modes for rendering."""

    FULL_SIM_STATE = 0  # Top-down orthographic view of full simulation
    BEV_AGENT_OBS = 1  # Bird's eye view centered on agent observation
    AGENT_PERSPECTIVE = 2  # Third-person chase camera following agent


@dataclass
class RenderContext:
    """Enables rendering inside rollout_loop.

    Attributes:
        view_mode: RenderView enum value passed to driver.render().
        env_id: which sub-env in the vecenv to record from (default 0).
        draw_traces: whether to draw trajectory traces.
        video_basename: full mp4 basename (without ".mp4"). Set once before
            the first render via driver.set_video_suffix. Caller is
            responsible for making this unique across renders.
    """

    view_mode: RenderView
    env_id: int = 0
    draw_traces: bool = True
    video_basename: str = "render"


def rollout_loop(
    policy,
    env,
    device,
    use_rnn: bool,
    max_steps: Optional[int] = None,
    render_ctx: Optional[RenderContext] = None,
):
    """Run a single policy rollout in a Drive vecenv.

    This function handles policy inference in Python/PyTorch, making it work
    with ANY model architecture (LSTM, Transformer, etc.).

    Args:
        policy: the policy to run. Caller is responsible for calling .eval().
        env: a PufferEnv-compatible vecenv wrapping one or more Drive sub-envs.
        device: torch device for observation / state tensors.
        use_rnn: whether to allocate and carry LSTM hidden state.
        max_steps: loop iteration cap. Defaults to env.driver_env.scenario_length.
        render_ctx: if set, render the specified env/view every step before
            sampling actions. Filename suffix is applied via set_video_suffix.

    Returns:
        The last info returned by env.step().
    """
    driver = env.driver_env

    # Handle population play mode - only ego agents are controlled by the policy
    population_play = getattr(driver, "population_play", False)
    if population_play:
        num_ego_agents = driver.num_ego_agents
        ego_ids = driver.ego_ids
        print(f"[rollout] Population play mode: {num_ego_agents} ego agents, {driver.num_co_players} co-players")
    else:
        num_ego_agents = env.observation_space.shape[0]
        ego_ids = None

    # Set full video basename before the first render call
    if render_ctx is not None:
        driver.set_video_suffix(render_ctx.video_basename, env_id=render_ctx.env_id)

    obs, _ = env.reset()

    # Initialize recurrent state based on policy type
    # Note: state is for EGO agents only, not co-players
    state = {}
    if use_rnn:
        # Check if this is a Transformer or LSTM policy
        is_transformer = hasattr(policy, "transformer") or hasattr(policy, "horizon")

        if is_transformer:
            # Transformer handles its own state initialization in forward_eval
            # when transformer_context is missing, so we just pass empty state
            state = {}
        else:
            # LSTM policy - initialize h and c states for ego agents only
            if hasattr(policy, "hidden_size"):
                hidden_size = policy.hidden_size
            else:
                hidden_size = 128  # default
            state = dict(
                lstm_h=torch.zeros(num_ego_agents, hidden_size, device=device),
                lstm_c=torch.zeros(num_ego_agents, hidden_size, device=device),
            )

    # Default max_steps:
    #  - non-trial: one scenario_length (matches single-episode video budget).
    #  - GOAL_TRIAL: a full episode budget = max_trials * per_trial_timeout
    #    (which auto-link sets to k_scenarios * scenario_length). Without
    #    this, the render would cut off after one scenario_length, showing
    #    only the first trial of an adaptive episode — the whole point of
    #    trial mode is to see adaptation ACROSS trials in one video.
    if max_steps is None:
        goal_behavior = int(getattr(driver, "goal_behavior", 0))
        if goal_behavior == 3:
            max_trials = int(getattr(driver, "max_trials_per_episode", 2))
            per_trial = int(getattr(driver, "per_trial_timeout", 0) or 0)
            if per_trial <= 0:
                per_trial = int(getattr(driver, "scenario_length", 91))
            max_steps = max_trials * per_trial
        else:
            max_steps = getattr(driver, "scenario_length", 91)

    # Per-trial annotation state. Under GOAL_TRIAL we read driver.trial_ended_this_step
    # after each step and bump a per-agent trial counter so the visualizer (or
    # downstream caller) knows which trial each frame belongs to.
    is_trial_mode = int(getattr(driver, "goal_behavior", 0)) == 3
    n_agents_for_trial = getattr(driver, "num_ego_agents", None) or env.observation_space.shape[0]
    trial_idx = np.zeros(n_agents_for_trial, dtype=np.int32) if is_trial_mode else None
    trial_starts = []  # list of (step, agent_idx, new_trial_idx) — useful for video chapter markers
    last_print_step = -1

    info = []
    for step in range(max_steps):
        if step % 30 == 0 and step != last_print_step:
            trial_suffix = f" trial=mean_{float(trial_idx.mean()):.1f}" if is_trial_mode else ""
            print(f"[Python Render] Step {step}/{max_steps}{trial_suffix}", flush=True)
            last_print_step = step
        # Render BEFORE the step so each frame shows the state the policy was
        # conditioned on.
        if render_ctx is not None:
            driver.render(
                view_mode=render_ctx.view_mode,
                draw_traces=render_ctx.draw_traces,
                env_id=render_ctx.env_id,
            )

        with torch.no_grad():
            # In population play, only pass ego observations to the policy
            if population_play:
                ego_obs = obs[ego_ids]
                ob_t = torch.as_tensor(ego_obs).to(device)
            else:
                ob_t = torch.as_tensor(obs).to(device)

            logits, _ = policy.forward_eval(ob_t, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)

            # Reshape actions to match expected format
            if population_play:
                # In population play, policy outputs actions for ego agents only
                # env.action_space.shape is (total_agents, action_dim), we need (num_ego_agents, action_dim)
                action_dim = env.action_space.shape[1] if len(env.action_space.shape) > 1 else 1
                action_np = action.cpu().numpy().reshape(num_ego_agents, action_dim)
            else:
                action_np = action.cpu().numpy().reshape(env.action_space.shape)

        # Clip continuous actions to the valid range
        if isinstance(logits, torch.distributions.Normal):
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        obs, _, terms, truncs, info = env.step(action_np)

        # Per-trial bookkeeping. trial_ended_this_step is per-agent — when it
        # fires we know that agent just started a new trial on the next step,
        # so bump its trial_idx. trial_starts collects (step, agent, new_idx)
        # tuples that callers can use to overlay trial boundaries on the video.
        if is_trial_mode and trial_idx is not None:
            te = np.asarray(driver.trial_ended_this_step, dtype=bool)
            if population_play and ego_ids is not None:
                te = te[ego_ids] if te.shape[0] == env.observation_space.shape[0] else te
            te = te[: len(trial_idx)]
            if te.any():
                trial_idx[te] += 1
                for a in np.where(te)[0]:
                    trial_starts.append((step + 1, int(a), int(trial_idx[a])))

        # Break conditions:
        #  - non-trial: truncs.all() fires at scenario boundary (env auto-reset path).
        #  - GOAL_TRIAL: truncs fires on every trial boundary (it now mirrors
        #    trial_ended_this_step — see drive.py.step). truncs.all() would
        #    fire whenever all agents end a trial at the same tick, which is
        #    NOT an episode boundary. Use terminals.all() instead so we
        #    render the full multi-trial episode.
        if is_trial_mode:
            if np.asarray(terms).all():
                break
        else:
            if truncs.all():
                break

    # Stash trial_starts on the info dict for downstream consumers (renderer
    # overlays, video chapter markers). Doesn't change existing info contract.
    if is_trial_mode and isinstance(info, list):
        info.append({"_trial_starts": trial_starts, "_final_trial_idx": trial_idx.tolist()})

    return info
