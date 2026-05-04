import numpy as np
import gymnasium
import json
import struct
import os
from enum import IntEnum
import pufferlib
from pufferlib.ocean.drive import binding
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class RenderView(IntEnum):
    """View modes for rendering."""

    FULL_SIM_STATE = 0  # Top-down orthographic view of full simulation
    BEV_AGENT_OBS = 1  # Bird's eye view centered on agent observation
    AGENT_PERSPECTIVE = 2  # Third-person chase camera following agent


class Drive(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        reward_lane_align=0.0,  # GIGAFLOW lane alignment reward (0 = disabled)
        reward_vel_align=1.0,  # Velocity alignment coefficient for lane reward
        goal_behavior=0,
        goal_target_distance=10.0,
        goal_radius=2.0,
        goal_speed=20.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        scenario_length=None,
        episode_length=None,
        termination_mode=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        max_controlled_agents=-1,
        buf=None,
        seed=1,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        k_scenarios=1,
        adaptive_driving_agent=False,
        ini_file="pufferlib/config/ocean/drive.ini",
        conditioning={},  # ego conditioning
        co_player_enabled=False,
        num_ego_agents=512,
        co_player_policy={},
        map_dir="resources/drive/binaries/training",
        use_all_maps=False,
        report_all_scenarios=False,
        map_seed=None,
        external_co_player_actions=False,
        worker_idx=0,
        co_player_conditioning_shm=None,
        map_rand_per_scenario=False,
        condition_rand_per_scenario=False,
        entropy_curriculum_enabled=False,
        entropy_curriculum_episodes_start=0,
        k_eff_curriculum_enabled=False,
        k_eff_curriculum_episodes_per_stage=30,
        ego_is_oracle=False,
        reward_only_last_scenario=False,
    ):
        # env
        self.dt = dt
        # Convert render_mode string to integer constant
        if render_mode is None or render_mode == 0:
            self._render_mode_int = binding.RENDER_OFF
        elif render_mode == 1 or render_mode == "headless":
            self._render_mode_int = binding.RENDER_HEADLESS
        elif render_mode == 2 or render_mode == "window" or render_mode == "human":
            self._render_mode_int = binding.RENDER_WINDOW
        else:
            self._render_mode_int = binding.RENDER_OFF
        self.render_mode = render_mode
        self.report_all_scenarios = report_all_scenarios
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.reward_lane_align = reward_lane_align
        self.reward_vel_align = reward_vel_align
        self.goal_radius = goal_radius
        self.goal_speed = goal_speed
        self.goal_behavior = goal_behavior
        self.goal_target_distance = goal_target_distance
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.human_agent_idx = human_agent_idx
        self.scenario_length = scenario_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.ini_file = ini_file
        self.use_all_maps = use_all_maps
        self.map_seed = map_seed

        if episode_length != None:
            self.scenario_length = episode_length
        # Only set episode_length if not already set (adaptive.py sets it before calling super())
        if not hasattr(self, "episode_length"):
            self.episode_length = self.scenario_length

        # Adaptive driving agent setup
        self.adaptive_driving_agent = int(adaptive_driving_agent)
        self.k_scenarios = int(k_scenarios)
        self.current_scenario = 0
        self.scenario_metrics = []  # List to store metrics for each scenario
        self.current_scenario_infos = []  # Accumulate infos for current scenario

        # Main policy conditioning setup
        self.conditioning = conditioning

        self.condition_type = self.conditioning.get("type", "none")
        self.reward_conditioned = self.condition_type in ("reward", "all")
        self.entropy_conditioned = self.condition_type in ("entropy", "all")
        self.discount_conditioned = self.condition_type in ("discount", "all")

        self.collision_weight_lb = (
            self.conditioning.get("collision_weight_lb", reward_vehicle_collision)
            if self.reward_conditioned
            else reward_vehicle_collision
        )
        self.collision_weight_ub = (
            self.conditioning.get("collision_weight_ub", reward_vehicle_collision)
            if self.reward_conditioned
            else reward_vehicle_collision
        )
        self.offroad_weight_lb = (
            self.conditioning.get("offroad_weight_lb", reward_offroad_collision)
            if self.reward_conditioned
            else reward_offroad_collision
        )
        self.offroad_weight_ub = (
            self.conditioning.get("offroad_weight_ub", reward_offroad_collision)
            if self.reward_conditioned
            else reward_offroad_collision
        )
        self.goal_weight_lb = (
            self.conditioning.get("goal_weight_lb", reward_goal) if self.reward_conditioned else reward_goal
        )
        self.goal_weight_ub = (
            self.conditioning.get("goal_weight_ub", reward_goal) if self.reward_conditioned else reward_goal
        )
        self.entropy_weight_lb = self.conditioning.get("entropy_weight_lb", 0.001)
        self.entropy_weight_ub = self.conditioning.get("entropy_weight_ub", 0.001)
        self.discount_weight_lb = self.conditioning.get("discount_weight_lb", 0.98)
        self.discount_weight_ub = self.conditioning.get("discount_weight_ub", 0.98)

        conditioning_dims = (
            (3 if self.reward_conditioned else 0)
            + (1 if self.entropy_conditioned else 0)
            + (1 if self.discount_conditioned else 0)
        )
        self.dynamics_model = dynamics_model

        # Observation space calculation
        self.ego_features = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(
            dynamics_model
        )

        self.ego_features += conditioning_dims

        # Extract observation shapes from constants
        # These need to be defined in C, since they determine the shape of the arrays
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = binding.MAX_AGENTS - 1
        self.partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES

        self.num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        # Co-player policy setup
        self.population_play = co_player_enabled
        self.num_agents = num_agents
        self.num_ego_agents = num_ego_agents if self.population_play else num_agents
        # When True, co-player actions are filled into self.actions[co_player_ids]
        # by the *main* process (centralized GPU inference). Worker's step()
        # then skips the local CPU forward in get_co_player_actions().
        self.external_co_player_actions = bool(external_co_player_actions)
        # When True (and adaptive_drive with k>1), at every scenario boundary
        # we re-init the C envs with FRESH map_ids (and freshly sampled co-
        # player conditioning). The agents are spawned on a brand-new map
        # while the EGO POLICY's K/V cache (held in main / pufferl) is NOT
        # touched — so past-scenario context is the only stable signal that
        # carries across scenarios. This is the experimental setup that
        # actually exercises in-context adaptation.
        self.map_rand_per_scenario = bool(map_rand_per_scenario)
        # When True (only meaningful for k_scenarios > 1, with co_player_enabled):
        # at every scenario boundary within an episode, partner conditioning is
        # re-sampled from the configured ranges. The same partner POLICY weights
        # are used, but the partner's effective behavior changes per scenario
        # because conditioning shifts. This gives the ego policy a meaningful
        # latent variable (partner type) to encode in its K/V cache without
        # introducing the agent-identity misalignment that map_rand causes.
        # Independent of map_rand_per_scenario.
        self.condition_rand_per_scenario = bool(condition_rand_per_scenario)
        # When True, partner's entropy_weight_ub is annealed up over training:
        # the user-passed co_player_entropy_weight_ub is treated as the FINAL
        # value, and a 4-stage schedule scales it 0.05 → 0.20 → 0.50 → 1.0 of
        # the final, advancing every 30 episodes per worker. The other
        # conditioning dims (collision/offroad/discount) sample at full range
        # throughout. Reason: we observed ada_delta peaking early in training
        # then drifting toward 0 as scores saturate; the curriculum keeps the
        # task in an informative-difficulty regime for longer.
        self.entropy_curriculum_enabled = bool(entropy_curriculum_enabled)
        # When resuming from a checkpoint of a curriculum run, the per-env
        # episode counter isn't part of the model state — pass the original
        # run's ending episode count here so the curriculum picks up at the
        # right stage instead of restarting from stage 0. Each worker still
        # advances its own counter from this starting value.
        self._entropy_curriculum_episodes_seen = int(entropy_curriculum_episodes_start)
        self._pending_entropy_log = None
        self._entropy_curriculum_final_ub = None  # set lazily once we know co_player_entropy_weight_ub
        # When True, the ego's K/V cache is reset at SOME within-episode
        # scenario boundaries based on the curriculum stage. K_max is the
        # configured `k_scenarios`; the curriculum has 3 stages of
        # `k_eff_curriculum_episodes_per_stage` episodes each:
        #   Stage 0: k_eff=1     (reset at every within-episode boundary)
        #   Stage 1: k_eff=2     (reset at boundaries where current_scenario%2==0)
        #   Stage 2: k_eff=k_max (no within-episode resets)
        # Implementation: at the boundary that should reset, we set
        # truncations[ego_ids]=1 and terminals[ego_ids]=1 for the current
        # step. pufferl picks up done_mask=t+d to reset transformer_position
        # at eval time; create_episode_mask uses terminals to block
        # cross-boundary attention during training. K_max=4 with stages
        # k_eff∈{1,2,4} gives clean splits (boundaries 1,2,3 → reset {all},
        # {middle only}, {none}). For other K_max, only k_eff=1 and k_max
        # produce uniform stages.
        self.k_eff_curriculum_enabled = bool(k_eff_curriculum_enabled)
        self.k_eff_curriculum_episodes_per_stage = int(k_eff_curriculum_episodes_per_stage)
        self._k_eff_curriculum_episodes_seen = 0
        self._pending_k_eff_log = None
        # When True, _reinit_envs_with_new_maps() donates env[0]->client to a
        # C-side global before vec_close and re-attaches it to the new env[0]
        # afterwards. This keeps the raylib window + ffmpeg pipe alive across
        # the swap (raylib's CloseWindow → InitWindow cycle segfaults under
        # xvfb), so a single mp4 captures all k scenarios with the maps
        # rotating mid-stream. Set True only on the single-env render driver.
        self._render_keep_client_on_swap = False
        self.worker_idx = int(worker_idx)
        # SHM view (numpy) of the per-worker conditioning slice. Env writes
        # sampled conditioning here so the main process can read it before
        # running the centralized co-player forward. None when conditioning
        # is disabled or when running the per-worker CPU path.
        self.co_player_conditioning_shm = co_player_conditioning_shm

        # Co-player conditioning setup
        self.co_player_conditioning = co_player_policy.get("conditioning")
        if self.co_player_conditioning:
            self.co_player_condition_type = self.co_player_conditioning.get("type")

            self.co_player_reward_conditioned = self.co_player_condition_type in ("reward", "all")
            self.co_player_entropy_conditioned = self.co_player_condition_type in ("entropy", "all")
            self.co_player_discount_conditioned = self.co_player_condition_type in ("discount", "all")

            self.co_player_collision_weight_lb = self.co_player_conditioning.get("collision_weight_lb", -0.5)
            self.co_player_collision_weight_ub = self.co_player_conditioning.get("collision_weight_ub", -0.5)
            self.co_player_offroad_weight_lb = self.co_player_conditioning.get("offroad_weight_lb", -0.2)
            self.co_player_offroad_weight_ub = self.co_player_conditioning.get("offroad_weight_ub", -0.2)
            self.co_player_goal_weight_lb = self.co_player_conditioning.get("goal_weight_lb", 1.0)
            self.co_player_goal_weight_ub = self.co_player_conditioning.get("goal_weight_ub", 1.0)
            self.co_player_entropy_weight_lb = self.co_player_conditioning.get("entropy_weight_lb", 0.001)
            self.co_player_entropy_weight_ub = self.co_player_conditioning.get("entropy_weight_ub", 0.001)
            self.co_player_discount_weight_lb = self.co_player_conditioning.get("discount_weight_lb", 0.98)
            self.co_player_discount_weight_ub = self.co_player_conditioning.get("discount_weight_ub", 0.98)

        # ----- Ego oracle (NEW, isolated machinery) -----
        # When True, the ego's obs gets the partner's per-env conditioning
        # vector appended at the END (after road_obs). Implementation:
        #   1. Allocate a private `_c_observations` buffer the C side writes
        #      into (sized to the C's expected obs_dim, no oracle slots).
        #   2. The pufferl-facing `self.observations` buffer is sized
        #      bigger (`+ oracle_dims`); each step we copy the C buffer
        #      into the first part and write `_oracle_obs_per_env[env]`
        #      into the trailing oracle slots for every ego row.
        # No changes to [env.conditioning], pufferl, or reward — the
        # oracle slots are pure obs signal that only the policy reads.
        self.ego_is_oracle = bool(ego_is_oracle)
        self.reward_only_last_scenario = bool(reward_only_last_scenario)
        if self.reward_only_last_scenario and not self.adaptive_driving_agent:
            raise ValueError("reward_only_last_scenario=True requires adaptive_driving_agent=True (k_scenarios > 1).")
        if self.ego_is_oracle:
            # Determine the partner's conditioning dim count (== oracle width).
            ct = self.co_player_condition_type
            if ct is None or ct == "none":
                raise ValueError(
                    "ego_is_oracle=True requires co-player conditioning to be "
                    "enabled (co_player_policy.conditioning.type != 'none')."
                )
            self._oracle_dims = (
                (3 if self.co_player_reward_conditioned else 0)
                + (1 if self.co_player_entropy_conditioned else 0)
                + (1 if self.co_player_discount_conditioned else 0)
            )
            if self._oracle_dims == 0:
                raise ValueError("ego_is_oracle=True but partner conditioning resolved to 0 dims.")
            # Grow obs space so pufferl allocates a buffer wide enough to
            # hold the appended oracle slots (placed AFTER road_obs, at
            # offset = num_obs - oracle_dims). C still writes the smaller
            # part into its own private `_c_observations` buffer.
            self.num_obs += self._oracle_dims
            self.single_observation_space = gymnasium.spaces.Box(
                low=-1, high=1, shape=(self.num_obs,), dtype=np.float32
            )
            # Per-env oracle vector. Filled at every _set_co_player_conditioning
            # call from a copy of `self.env_conditioning`. We don't allocate
            # `_c_observations` or `_ego_env_indices` here — they need
            # `num_envs`/`num_agents` which aren't known until later.
            self._oracle_obs_per_env = None
            self._c_observations = None
            self._ego_env_indices = None
        else:
            self._oracle_dims = 0

        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.map_dir = map_dir

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_wosac":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        else:
            raise ValueError(
                f"control_mode must be one of 'control_vehicles', 'control_tracks_to_predict', or 'control_agents'. Got: {self.control_mode_str}"
            )
        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        else:
            raise ValueError(
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )

        if action_type == "discrete":
            if dynamics_model == "classic":
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check if resources directory exists (check map_001 since some datasets start at 001)
        binary_path = f"{map_dir}/map_001.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Required file {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len([name for name in os.listdir(map_dir) if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to {map_dir}."
            )
        if self.population_play:
            if self.num_ego_agents > num_agents:
                raise ValueError(
                    f"num ego agents ({self.num_ego_agents}) exceeds the number of total agents ({num_agents}))"
                )
            if self.condition_type != "none" and self.co_player_condition_type != "none":
                raise NotImplementedError("Only one agent can be conditioned at once")

        self.max_controlled_agents = int(max_controlled_agents)

        self._set_env_variables()

        if self.population_play:
            self.co_player_policy_name = co_player_policy.get("policy_name")
            self.co_player_rnn_name = co_player_policy.get("rnn_name")
            if self.external_co_player_actions:
                # Main owns the policy + state on GPU; worker only needs the
                # action slots (co_player_ids) to be filled via shared memory
                # before vec_step. Skip the per-worker CPU model entirely.
                self.co_player_policy = None
                self.co_player_device = None
            else:
                self.co_player_policy = co_player_policy.get("co_player_policy_func")
                # Co-player runs in forked subprocess - must stay on CPU
                # (CUDA doesn't work with fork)
                self.co_player_device = torch.device("cpu")
                self._set_co_player_state()

        super().__init__(buf=buf)

        # `trial_ended_this_step`: per-agent flag set by C in c_step under
        # goal_behavior=GOAL_TRIAL (=3) when a trial ends (goal-reach OR
        # per-trial timeout). Distinct from `terminals`, which fires only
        # at the EPISODE boundary (after max_trials_per_episode trials).
        # Python-owned 1-byte buffer; C reads the pointer set in env_init.
        self.trial_ended_this_step = np.zeros(self.num_agents, dtype=bool)

        if self.population_play:
            self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_ego_agents)
            co_player_atn_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_co_players)
            if isinstance(self.single_action_space, pufferlib.spaces.Box):
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=co_player_atn_space.dtype)
            else:
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=np.int32)

        # Allocate the private C-only obs buffer + ego→env index map (oracle
        # path only). C writes into `_c_observations` (no oracle slots);
        # we copy + append into `self.observations` (which has oracle slots)
        # every step/reset. `_oracle_obs_per_env` was filled by
        # `_set_co_player_conditioning` during the prior `_set_env_variables`
        # call (which always fires here because oracle requires co-player
        # conditioning to be on).
        if self.ego_is_oracle:
            self._c_obs_dim = self.num_obs - self._oracle_dims
            self._c_observations = np.zeros((self.num_agents, self._c_obs_dim), dtype=np.float32)
            self._rebuild_ego_env_indices()

        env_ids = []
        for i in range(self.num_envs):
            cur = self.agent_offsets[i]
            nxt = self.agent_offsets[i + 1]
            # Oracle: hand C its own private obs slice (smaller, no oracle
            # slots). Otherwise C uses the pufferl-provided buffer directly.
            obs_slice_for_c = self._c_observations[cur:nxt] if self.ego_is_oracle else self.observations[cur:nxt]
            env_id = binding.env_init(
                obs_slice_for_c,
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=human_agent_idx,
                dynamics_model=dynamics_model,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                reward_lane_align=self.reward_lane_align,
                reward_vel_align=self.reward_vel_align,
                goal_radius=goal_radius,
                goal_speed=goal_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                map_id=self.map_ids[i],
                max_agents=nxt - cur,
                ini_file=self.ini_file,
                population_play=self.population_play,
                num_co_players=len(self.local_co_player_ids[i]),
                co_player_ids=self.local_co_player_ids[i],
                ego_agent_ids=self.local_ego_ids[i],
                num_ego_agents=len(self.local_ego_ids[i]),
                init_steps=init_steps,
                use_rc=self.reward_conditioned,
                use_ec=self.entropy_conditioned,
                use_dc=self.discount_conditioned,
                collision_weight_lb=self.collision_weight_lb,
                collision_weight_ub=self.collision_weight_ub,
                offroad_weight_lb=self.offroad_weight_lb,
                offroad_weight_ub=self.offroad_weight_ub,
                goal_weight_lb=self.goal_weight_lb,
                goal_weight_ub=self.goal_weight_ub,
                entropy_weight_lb=self.entropy_weight_lb,
                entropy_weight_ub=self.entropy_weight_ub,
                discount_weight_lb=self.discount_weight_lb,
                discount_weight_ub=self.discount_weight_ub,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=map_dir,
                render_mode=self._render_mode_int,
                trial_ended_this_step=self.trial_ended_this_step[cur:nxt],
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        # Oracle: copy C obs into pufferl buffer + write oracle slots.
        self._refresh_ego_oracle_obs()
        info = []
        if self.population_play:
            info.append(self.ego_ids)
            if self.external_co_player_actions:
                # Pass the real co_player_ids so main does not have to
                # guess by complement (which would include padding slots
                # and pollute the shared KV cache).
                info.append({"_external_co_player_ids": self.co_player_ids})
                # Tell main to drop the K/V cache, mirroring OFF's
                # `_reset_co_player_state()` here. Initial conditioning
                # was written to SHM in `_set_env_variables` and stays
                # valid across the episode (matching OFF, which doesn't
                # re-sample at reset either).
                info.append({"_external_reset_co_cache": True})
            else:
                self._reset_co_player_state()
        self.tick = 0
        return self.observations, info

    def _set_env_variables(self):
        my_shared_tuple = binding.shared(
            map_dir=self.map_dir,
            num_agents=self.num_agents,
            num_maps=self.num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            population_play=self.population_play,
            num_ego_agents=self.num_ego_agents,
            goal_target_distance=self.goal_target_distance,
            use_all_maps=self.use_all_maps,
            map_seed=self.map_seed if self.map_seed is not None else -1,
        )

        if self.population_play:
            self.agent_offsets, self.map_ids, num_envs, ego_ids, co_player_ids = my_shared_tuple

            self.num_envs = num_envs

            self.ego_ids = [item for sublist in ego_ids for item in sublist]
            self.co_player_ids = [item for sublist in co_player_ids for item in sublist]

            all_agents = set(range(self.num_agents))
            ego_set = set(self.ego_ids)
            co_player_set = set(self.co_player_ids)
            self.num_ego_agents = len(self.ego_ids)
            self.num_co_players = len(self.co_player_ids)

            if ego_set & co_player_set:
                raise ValueError("Overlap between ego ids and co player ids")

            if ego_set | co_player_set != all_agents:
                raise ValueError("Missing agent ids")

            if self.num_ego_agents + self.num_co_players != self.num_agents:
                raise ValueError("Mismatch between number of ego/co players and number of agents")

            self.total_agents = self.num_co_players + self.num_ego_agents
            self.num_agents = self.total_agents

            # Build per-environment ID lists
            local_ego_ids = []
            for i in range(num_envs):
                if len(ego_ids[i]) > 0 and len(co_player_ids[i]) > 0:
                    min_id_in_world = min(ego_ids[i] + co_player_ids[i])
                elif len(ego_ids[i]) > 0:
                    min_id_in_world = min(ego_ids[i])
                elif len(co_player_ids[i]) > 0:
                    min_id_in_world = min(co_player_ids[i])
                else:
                    min_id_in_world = 0

                local_ego_ids.append([eid - min_id_in_world for eid in ego_ids[i]])

            local_co_player_ids = []
            for i in range(num_envs):
                if len(ego_ids[i]) > 0 and len(co_player_ids[i]) > 0:
                    min_id_in_world = min(ego_ids[i] + co_player_ids[i])
                elif len(ego_ids[i]) > 0:
                    min_id_in_world = min(ego_ids[i])
                elif len(co_player_ids[i]) > 0:
                    min_id_in_world = min(co_player_ids[i])
                else:
                    min_id_in_world = 0

                local_co_player_ids.append([cid - min_id_in_world for cid in co_player_ids[i]])

            self.local_co_player_ids = local_co_player_ids
            self.local_ego_ids = local_ego_ids
            if self.co_player_condition_type is not None and self.co_player_condition_type != "none":
                self._set_co_player_conditioning()

        else:
            self.agent_offsets, self.map_ids, self.num_envs = my_shared_tuple
            self.ego_ids = [i for i in range(self.agent_offsets[-1])]
            if len(self.ego_ids) != self.num_agents:
                print(
                    f"Warning: requested {self.num_agents} agents but maps contain {len(self.ego_ids)} valid agents. Adjusting.",
                    flush=True,
                )
                self.num_agents = len(self.ego_ids)
            self.local_co_player_ids = [[] for i in range(self.num_envs)]
            self.local_ego_ids = [[0] for i in range(self.num_envs)]

    def get_co_player_actions(self):
        with torch.no_grad():
            co_player_obs = self.observations[self.co_player_ids]
            # Add conditioning to co-player observations if needed
            if self.co_player_condition_type != "none":
                co_player_obs = self._add_co_player_conditioning(co_player_obs)

            # Convert directly to device for GPU acceleration
            co_player_obs = torch.as_tensor(co_player_obs, device=self.co_player_device)
            import sys

            sys.stdout.flush()  # Prevent multiprocessing deadlock
            logits, value = self.co_player_policy.forward_eval(co_player_obs, self.state)
            # Handle multi-discrete actions (logits is a tuple) vs single discrete (logits is tensor)
            if isinstance(logits, tuple):
                co_player_action = torch.cat([l.argmax(dim=-1, keepdim=True) for l in logits], dim=-1)
            else:
                co_player_action = logits.argmax(dim=-1)
            # Only this transfer is necessary
            co_player_action = co_player_action.cpu().numpy().reshape(self.co_player_actions.shape)
        return co_player_action

    def _set_co_player_state(self):
        with torch.no_grad():
            # Detect if co-player uses Transformer (has horizon) or LSTM
            self.co_player_is_transformer = hasattr(self.co_player_policy, "horizon")

            if self.co_player_is_transformer:
                # Transformer co-player uses streaming KV cache (see
                # TransformerWrapper.forward_eval). The cache is allocated
                # lazily inside forward_eval the first time it sees a state
                # without "k_cache". We initialize the position counter here
                # so reset_eval_state has something to zero on full reset.
                self.state = dict(
                    transformer_position=torch.zeros(1, dtype=torch.long, device=self.co_player_device),
                )
            else:
                self.state = dict(
                    lstm_h=torch.zeros(
                        self.num_co_players, self.co_player_policy.hidden_size, device=self.co_player_device
                    ),
                    lstm_c=torch.zeros(
                        self.num_co_players, self.co_player_policy.hidden_size, device=self.co_player_device
                    ),
                )

    def _reset_co_player_state(self, done_indices=None):
        """Reset LSTM/Transformer state for co-players whose episodes ended"""
        with torch.no_grad():
            if done_indices is None:
                # Reset all
                self._set_co_player_state()
            else:
                # Reset only specific co-players
                if self.co_player_is_transformer:
                    # Re-prime the KV cache for the done rows so that subsequent
                    # forward_eval calls behave as if those rows had a fresh
                    # zero hidden buffer (matches the original semantics of
                    # `state["transformer_context"][done_indices] = 0`).
                    self.co_player_policy.reset_eval_state(self.state, done_indices=done_indices)
                else:
                    self.state["lstm_h"][done_indices] = 0
                    self.state["lstm_c"][done_indices] = 0

    def _add_co_player_conditioning(self, observations):
        """Add pre-sampled conditioning variables to co-player observations"""
        if self.cached_conditioning_array.shape[1] == 0:  # No conditioning
            return observations

        # Early return if no co-players
        if self.total_co_players == 0:
            return observations

        # Validate observations shape (optional, can remove in production for speed)
        if observations.shape[0] != self.total_co_players:
            raise ValueError(f"Expected {self.total_co_players} observations, got {observations.shape[0]}")

        # Use dynamic base_ego_dim based on dynamics model
        base_ego_dim = binding.EGO_FEATURES_JERK if self.dynamics_model == "jerk" else binding.EGO_FEATURES_CLASSIC
        return np.concatenate(
            [observations[:, :base_ego_dim], self.cached_conditioning_array, observations[:, base_ego_dim:]], axis=1
        )

    def _rebuild_ego_env_indices(self):
        """Recompute self._ego_env_indices: for the k-th ego in self.ego_ids
        order, the env index it belongs to. Called at init and after every
        _set_env_variables (since map re-roll may change per-env ego counts).
        Oracle path only."""
        if not self.ego_is_oracle:
            return
        ego_env_ids = []
        if self.population_play:
            for env_idx, env_egos in enumerate(self.local_ego_ids):
                ego_env_ids.extend([env_idx] * len(env_egos))
        else:
            for env_idx in range(self.num_envs):
                cur = int(self.agent_offsets[env_idx])
                nxt = int(self.agent_offsets[env_idx + 1])
                ego_env_ids.extend([env_idx] * (nxt - cur))
        self._ego_env_indices = np.asarray(ego_env_ids, dtype=np.int64)

    def _refresh_ego_oracle_obs(self):
        """Copy C-side obs into the pufferl-facing buffer and write the
        per-env partner-conditioning vector into the trailing oracle
        slots for every ego row. No-op when oracle is off."""
        if not self.ego_is_oracle:
            return
        c_dim = self._c_obs_dim
        # Copy C output into the leading c_obs_dim columns. (Cannot slice
        # the assignment to a single np.copyto because the buffers were
        # allocated separately; numpy fast path is fine.)
        self.observations[:, :c_dim] = self._c_observations
        # Append partner conditioning to ego rows only. Non-ego rows keep
        # whatever was there (zeros from allocation; pufferl filters out
        # non-ego rows downstream anyway).
        if len(self.ego_ids) > 0:
            self.observations[self.ego_ids, c_dim:] = self._oracle_obs_per_env[self._ego_env_indices]

    def _current_k_eff(self):
        """Effective k for the ego's K/V cache horizon at the current
        curriculum stage. Returns k_scenarios (i.e. K_max) when the
        curriculum is disabled or has finished. Stages last
        `k_eff_curriculum_episodes_per_stage` episodes each:
            stage 0 → k_eff = 1
            stage 1 → k_eff = 2
            stage 2+ → k_eff = K_max
        """
        if not self.k_eff_curriculum_enabled:
            return self.k_scenarios
        n = self._k_eff_curriculum_episodes_seen
        s = self.k_eff_curriculum_episodes_per_stage
        if n < s:
            return 1
        elif n < 2 * s:
            return 2
        else:
            return self.k_scenarios

    def _k_eff_should_reset_at_current_boundary(self):
        """True when the just-crossed scenario boundary should cut the ego
        K/V cache under the current curriculum stage. Caller must already
        have ensured this is a within-episode boundary (current_scenario != 0
        after the modulo increment)."""
        k_eff = self._current_k_eff()
        return self.current_scenario % k_eff == 0

    def _set_co_player_conditioning(self):
        """Sample and store conditioning values for each environment and update all caches"""
        # Entropy curriculum: scale entropy_ub based on episodes seen so far.
        # Schedule: stage 0 = 0.05*final, stage 1 = 0.20*final, stage 2 = 0.50*final,
        # stage 3 = 1.00*final. Each stage = 30 episodes per worker (≈30 epochs
        # given ~1 episode/epoch with our nw=32 nv=32 setup).
        if self.entropy_curriculum_enabled and self.co_player_entropy_conditioned:
            if self._entropy_curriculum_final_ub is None:
                self._entropy_curriculum_final_ub = self.co_player_entropy_weight_ub
            n = self._entropy_curriculum_episodes_seen
            if n < 30:
                ratio = 0.05
            elif n < 60:
                ratio = 0.20
            elif n < 90:
                ratio = 0.50
            else:
                ratio = 1.00
            self.co_player_entropy_weight_ub = ratio * self._entropy_curriculum_final_ub
            self._entropy_curriculum_episodes_seen += 1

        # Update co-player counts and indices
        self.num_co_players_per_env = np.array([len(ids) for ids in self.local_co_player_ids], dtype=np.int32)
        self.total_co_players = self.num_co_players_per_env.sum()

        # Pre-compute env_indices
        if self.total_co_players > 0:
            self.co_player_env_indices = np.repeat(
                np.arange(self.num_envs, dtype=np.int32), self.num_co_players_per_env
            )
        else:
            self.co_player_env_indices = np.array([], dtype=np.int32)

        # Sample conditioning values
        conditioning_dims = []

        if self.co_player_reward_conditioned:
            conditioning_dims.extend(
                [
                    (self.co_player_collision_weight_lb, self.co_player_collision_weight_ub),
                    (self.co_player_offroad_weight_lb, self.co_player_offroad_weight_ub),
                    (self.co_player_goal_weight_lb, self.co_player_goal_weight_ub),
                ]
            )

        if self.co_player_entropy_conditioned:
            conditioning_dims.append((self.co_player_entropy_weight_lb, self.co_player_entropy_weight_ub))

        if self.co_player_discount_conditioned:
            conditioning_dims.append((self.co_player_discount_weight_lb, self.co_player_discount_weight_ub))

        if not conditioning_dims:
            self.env_conditioning = np.empty((self.num_envs, 0), dtype=np.float32)
            self.cached_conditioning_array = np.empty((self.total_co_players, 0), dtype=np.float32)
        else:
            # Vectorized sampling
            lbs = np.array([lb for lb, ub in conditioning_dims], dtype=np.float32)
            ubs = np.array([ub for lb, ub in conditioning_dims], dtype=np.float32)

            random_values = np.random.uniform(size=(self.num_envs, len(conditioning_dims))).astype(np.float32)
            self.env_conditioning = lbs + random_values * (ubs - lbs)

            # Cache the conditioning array for co-players
            if self.total_co_players > 0:
                self.cached_conditioning_array = self.env_conditioning[self.co_player_env_indices]
            else:
                self.cached_conditioning_array = np.empty((0, len(conditioning_dims)), dtype=np.float32)

        # Oracle: sync per-env oracle vector with the freshly sampled
        # partner conditioning. num_envs can change across
        # _reinit_envs_with_new_maps (some maps yield no valid agents and
        # get dropped C-side), so just take a fresh copy with the current
        # shape rather than a fixed-size in-place write. Width invariant
        # is checked by the assertion below.
        if self.ego_is_oracle:
            assert self.env_conditioning.shape[1] == self._oracle_dims, (
                f"oracle width mismatch: env_conditioning has "
                f"{self.env_conditioning.shape[1]} dims, oracle expects {self._oracle_dims}"
            )
            self._oracle_obs_per_env = self.env_conditioning.copy()

        # Stash sampled entropy stats for wandb. Index of the entropy column
        # within env_conditioning depends on which dims are active above:
        #   reward(3) -> [collision, offroad, goal] then entropy then discount
        if self.co_player_entropy_conditioned and self.env_conditioning.shape[1] > 0:
            entropy_col = 3 if self.co_player_reward_conditioned else 0
            sampled = self.env_conditioning[:, entropy_col]
            self._pending_entropy_log = {
                "co_player/entropy_weight_ub": float(self.co_player_entropy_weight_ub),
                "co_player/entropy_sampled_mean": float(sampled.mean()),
                "co_player/entropy_sampled_min": float(sampled.min()),
                "co_player/entropy_sampled_max": float(sampled.max()),
            }
            if self.entropy_curriculum_enabled:
                self._pending_entropy_log["co_player/entropy_curriculum_episodes"] = int(
                    self._entropy_curriculum_episodes_seen
                )

        # Mirror the freshly-sampled conditioning into the shared-memory
        # buffer so the main process (centralized co-player on GPU) sees the
        # latest values before its next forward pass. SHM rows beyond
        # `total_co_players` are left at whatever value they previously held;
        # main only reads rows for active co-players.
        if (
            self.co_player_conditioning_shm is not None
            and self.cached_conditioning_array.shape[1] > 0
            and self.total_co_players > 0
        ):
            shm = self.co_player_conditioning_shm
            n = min(self.total_co_players, shm.shape[0])
            shm[:n, :] = self.cached_conditioning_array[:n, :]

    def _reinit_envs_with_new_maps(self):
        """Close + recreate C envs with fresh map_ids.

        Called at episode resample boundary and (when
        `map_rand_per_scenario=True`) at scenario boundaries. Note: this
        sets `self.terminals[:] = 1`, which pufferl uses to wipe the ego
        K/V cache — so `map_rand_per_scenario=True` is currently broken
        as an ICL probe (cache + GAE both truncate at the boundary).
        """
        if self._render_keep_client_on_swap:
            binding.vec_donate_client(self.c_envs)
        binding.vec_close(self.c_envs)
        self._set_env_variables()
        env_ids = []
        seed = np.random.randint(0, 2**32 - 1)
        for i in range(self.num_envs):
            cur = self.agent_offsets[i]
            nxt = self.agent_offsets[i + 1]
            obs_slice_for_c = self._c_observations[cur:nxt] if self.ego_is_oracle else self.observations[cur:nxt]
            env_id = binding.env_init(
                obs_slice_for_c,
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=self.human_agent_idx,
                dynamics_model=self.dynamics_model,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                goal_radius=self.goal_radius,
                goal_behavior=self.goal_behavior,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                reward_goal=self.reward_goal,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                reward_lane_align=self.reward_lane_align,
                reward_vel_align=self.reward_vel_align,
                goal_speed=self.goal_speed,
                goal_target_distance=self.goal_target_distance,
                dt=self.dt,
                scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                max_controlled_agents=self.max_controlled_agents,
                map_id=self.map_ids[i],
                use_rc=self.reward_conditioned,
                use_ec=self.entropy_conditioned,
                use_dc=self.discount_conditioned,
                collision_weight_lb=self.collision_weight_lb,
                collision_weight_ub=self.collision_weight_ub,
                offroad_weight_lb=self.offroad_weight_lb,
                offroad_weight_ub=self.offroad_weight_ub,
                goal_weight_lb=self.goal_weight_lb,
                goal_weight_ub=self.goal_weight_ub,
                entropy_weight_lb=self.entropy_weight_lb,
                entropy_weight_ub=self.entropy_weight_ub,
                discount_weight_lb=self.discount_weight_lb,
                discount_weight_ub=self.discount_weight_ub,
                max_agents=nxt - cur,
                ini_file=self.ini_file,
                population_play=self.population_play,
                num_co_players=len(self.local_co_player_ids[i]),
                co_player_ids=self.local_co_player_ids[i],
                ego_agent_ids=self.local_ego_ids[i],
                num_ego_agents=len(self.local_ego_ids[i]),
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=self.map_dir,
                render_mode=self._render_mode_int,
                trial_ended_this_step=self.trial_ended_this_step[cur:nxt],
            )
            env_ids.append(env_id)
        self.c_envs = binding.vectorize(*env_ids)
        if self._render_keep_client_on_swap:
            binding.vec_adopt_client(self.c_envs)

        binding.vec_reset(self.c_envs, seed)
        # Oracle: per-env ego counts may have shifted with the new map IDs;
        # rebuild the ego→env index map and refresh obs.
        self._rebuild_ego_env_indices()
        self._refresh_ego_oracle_obs()
        self.terminals[:] = 1

    def _aggregate_scenario_metrics(self, scenario_infos):
        """Aggregate metrics from all infos collected during a scenario."""
        if not scenario_infos:
            return {}

        # Sum up all metrics
        aggregated = {}
        count = len(scenario_infos)

        for log in scenario_infos:
            for key, value in log.items():
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + value

        # Average by number of logs (metrics are already per-episode averages from vec_log)
        for key in aggregated:
            aggregated[key] = aggregated[key] / count if count > 0 else 0.0

        return aggregated

    def _compute_delta_metrics(self):
        """Compute delta metrics between first and last scenario."""
        if len(self.scenario_metrics) < 2:
            return {}

        first_metrics = self.scenario_metrics[0]
        last_metrics = self.scenario_metrics[-1]

        delta_metrics = {}

        # Compute deltas for key metrics
        metrics_to_track = [
            "score",
            "collision_rate",
            "offroad_rate",
            "completion_rate",
            "dnf_rate",
            "num_goals_reached",
            "lane_alignment_rate",
            "avg_displacement_error",
            "episode_return",
            "perf",
        ]

        for metric in metrics_to_track:
            if metric in first_metrics and metric in last_metrics:
                delta_key = f"ada_delta_{metric}"
                delta_metrics[delta_key] = last_metrics[metric] - first_metrics[metric]

        return delta_metrics

    def step(self, actions):
        self.terminals[:] = 0

        self.actions[self.ego_ids] = actions

        if self.population_play and not self.external_co_player_actions:
            co_player_actions = self.get_co_player_actions()
            self.actions[self.co_player_ids] = co_player_actions
        # When external_co_player_actions=True, the main process has already
        # written co-player actions into self.actions[co_player_ids] via the
        # shared-memory action buffer; nothing to do here.

        binding.vec_step(self.c_envs)
        if self.reward_only_last_scenario and self.current_scenario != self.k_scenarios - 1:
            self.rewards[:] = 0
        # Oracle: copy C obs into pufferl buffer + write oracle slots.
        self._refresh_ego_oracle_obs()

        self.tick += 1
        info = []

        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                if self.adaptive_driving_agent:
                    self.current_scenario_infos.append(log)
                    # For training: only report 0-shot (scenario 0) metrics
                    # For evaluation: report all scenarios when report_all_scenarios=True
                    if self.current_scenario == 0 or self.report_all_scenarios:
                        info.append(log)
                else:
                    # Non-adaptive mode: always append
                    info.append(log)

            # Surface the entropy bound + sampled distribution that
            # _set_co_player_conditioning stashed at the most recent reset.
            # Drained on emit so each fresh sampling gets logged exactly once.
            if self._pending_entropy_log is not None:
                info.append(self._pending_entropy_log)
                self._pending_entropy_log = None
            if self._pending_k_eff_log is not None:
                info.append(self._pending_k_eff_log)
                self._pending_k_eff_log = None

        if self.tick % self.scenario_length == 0:
            if self.adaptive_driving_agent and self.current_scenario_infos:
                scenario_log = self._aggregate_scenario_metrics(self.current_scenario_infos)
                scenario_log["scenario_id"] = self.current_scenario
                self.scenario_metrics.append(scenario_log)

                # Log metrics for all scenarios with scenario-specific prefixes
                prefixed_log = {
                    f"scenario_{self.current_scenario}_{k}": v for k, v in scenario_log.items() if k != "scenario_id"
                }
                info.append(prefixed_log)

                if self.current_scenario == self.k_scenarios - 1:
                    delta_metrics = self._compute_delta_metrics()
                    if delta_metrics:
                        info.append(delta_metrics)

                    self.scenario_metrics = []

                self.current_scenario_infos = []

            self.current_scenario = (self.current_scenario + 1) % self.k_scenarios

            # Reset coplayer LSTM/Transformer state at scenario boundary so
            # the partner behaves consistently. The OFF (per-worker) path
            # does not re-sample conditioning here — it sticks with the
            # values written to SHM at init/resample — so neither do we.
            if self.population_play:
                if self.external_co_player_actions:
                    # Signal main to drop the K/V cache so scenario N+1
                    # starts fresh, mirroring the per-worker OFF path's
                    # `_reset_co_player_state()` here.
                    info.append({"_external_reset_co_cache": True})
                else:
                    self._reset_co_player_state()

            # MAP ROTATION per scenario: re-init the C envs with new map_ids
            # while leaving the EGO POLICY's K/V cache (held in main) alone.
            # This forces the policy to actually USE its past-scenario context
            # because the current scene is genuinely new.
            if (
                self.adaptive_driving_agent
                and self.map_rand_per_scenario
                and self.current_scenario != 0  # we just incremented above; 0 means we already wrapped to next episode
            ):
                self._reinit_envs_with_new_maps()
            # PARTNER CONDITIONING ROTATION per scenario: resample the partner's
            # conditioning vector. The partner POLICY is unchanged but its
            # effective behavior shifts (e.g. high-entropy stochastic vs
            # low-entropy deterministic) — the ego must encode partner type
            # from s_0 observations. Skipped when map_rand fired above
            # (reinit already re-samples conditioning via _set_env_variables).
            elif (
                self.adaptive_driving_agent
                and self.condition_rand_per_scenario
                and self.population_play
                and self.current_scenario != 0
                and self.co_player_condition_type is not None
                and self.co_player_condition_type != "none"
            ):
                self._set_co_player_conditioning()

            # k_eff curriculum: at within-episode scenario boundaries, decide
            # whether to cut the ego's K/V cache. Setting truncations=1 (and
            # terminals=1) at this step makes pufferl drop the cache via
            # done_mask=t+d, and during training, create_episode_mask blocks
            # cross-boundary attention. current_scenario != 0 excludes the
            # episode boundary itself (which is reset by the normal episode-
            # done logic). Reset rule: cut if current_scenario % k_eff == 0.
            if (
                self.adaptive_driving_agent
                and self.current_scenario != 0
                and self._k_eff_should_reset_at_current_boundary()
            ):
                self.truncations[self.ego_ids] = 1
                self.terminals[self.ego_ids] = 1

        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            will_resample = 1
            if will_resample:
                # Log deltas before resampling if we're at the end of a cycle
                if self.adaptive_driving_agent and self.scenario_metrics:
                    delta_metrics = self._compute_delta_metrics()
                    if delta_metrics:
                        info.append(delta_metrics)
                    self.scenario_metrics = []
                    self.current_scenario_infos = []
                    self.current_scenario = 0

                # Advance k_eff curriculum once per real episode and stash
                # the new stage's k_eff for wandb. Done before reinit so the
                # log reflects the stage that the next episode will run at.
                if self.k_eff_curriculum_enabled:
                    self._k_eff_curriculum_episodes_seen += 1
                    self._pending_k_eff_log = {
                        "ego_curriculum/k_eff": int(self._current_k_eff()),
                        "ego_curriculum/episodes_seen": int(self._k_eff_curriculum_episodes_seen),
                    }

                self._reinit_envs_with_new_maps()

        if self.population_play:
            info.append(self.ego_ids)
            if self.external_co_player_actions:
                info.append({"_external_co_player_ids": self.co_player_ids})

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def get_global_agent_state(self):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id', 'length', 'width' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "length": np.zeros(num_agents, dtype=np.float32),
            "width": np.zeros(num_agents, dtype=np.float32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs,
            states["x"],
            states["y"],
            states["z"],
            states["heading"],
            states["id"],
            states["length"],
            states["width"],
        )

        return states

    def get_ground_truth_trajectories(self):
        """Get ground truth trajectories for all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'valid', 'id', 'scenario_id' containing numpy arrays.
        """
        num_agents = self.num_agents

        trajectories = {
            "x": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "valid": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.int32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "scenario_id": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_ground_truth_trajectories(
            self.c_envs,
            trajectories["x"],
            trajectories["y"],
            trajectories["z"],
            trajectories["heading"],
            trajectories["valid"],
            trajectories["id"],
            trajectories["scenario_id"],
        )

        for key in trajectories:
            trajectories[key] = trajectories[key][:, None]

        return trajectories

    def get_road_edge_polylines(self):
        """Get road edge polylines for all scenarios.

        Returns:
            dict with keys 'x', 'y', 'lengths', 'scenario_id' containing numpy arrays.
            x, y are flattened point coordinates; lengths indicates points per polyline.
        """
        num_polylines, total_points = binding.vec_get_road_edge_counts(self.c_envs)

        polylines = {
            "x": np.zeros(total_points, dtype=np.float32),
            "y": np.zeros(total_points, dtype=np.float32),
            "lengths": np.zeros(num_polylines, dtype=np.int32),
            "scenario_id": np.zeros(num_polylines, dtype=np.int32),
        }

        binding.vec_get_road_edge_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["scenario_id"],
        )

        return polylines

    def render(self, view_mode: int = 0, draw_traces: bool = True, env_id: int = 0):
        """Render the environment.

        Args:
            view_mode: View mode for rendering:
                0 = VIEW_MODE_SIM_STATE (top-down orthographic)
                1 = VIEW_MODE_BEV_AGENT_OBS (bird's eye view centered on agent)
                2 = VIEW_MODE_AGENT_PERSP (third-person chase camera)
            draw_traces: Whether to draw trajectory traces
            env_id: Which environment to render (default 0)
        """
        binding.vec_render(self.c_envs, int(view_mode), draw_traces, env_id, self.current_scenario, self.k_scenarios)

    def set_video_suffix(self, suffix: str, env_id: int = 0):
        """Set the suffix appended to the mp4 filename for headless rendering.

        Must be called before the first render() call of a rollout.
        E.g. set_video_suffix("_bev", env_id=0) -> {scenario_id}_bev.mp4

        Args:
            suffix: Suffix string to append to video filename
            env_id: Which environment to set suffix for (default 0)
        """
        binding.vec_set_video_suffix(self.c_envs, env_id, suffix)

    def close(self):
        binding.vec_close(self.c_envs)


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            if area < polyline_reduction_threshold and dist(point1, point3) <= max_segment_length:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def _to_int32(v, default=0):
    """Wrap an arbitrary integer into the signed int32 range using two's-complement
    semantics so struct.pack('i', ...) cannot overflow. nuPlan IDs and some type
    fields can exceed 2^31-1; this preserves the low 32 bits the way C would."""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    v &= 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return v


def save_map_binary(map_data, output_file, unique_map_id, trajectory_length=91):
    """Saves map data in a binary format readable by C.

    `trajectory_length` is how many frames per object/road to write. The
    C reader is parametric on the per-binary `array_size` header, so any
    value works. Default 91 matches the legacy WOMD/short-window setup;
    nuplan scenes go up to 201 frames so pass `trajectory_length=201`
    to capture the full data."""
    with open(output_file, "wb") as f:
        # Get metadata
        metadata = map_data.get("metadata", {})
        sdc_track_index = metadata.get("sdc_track_index", -1)  # -1 as default if not found
        tracks_to_predict = metadata.get("tracks_to_predict", [])

        # Write sdc_track_index
        f.write(struct.pack("i", _to_int32(sdc_track_index, -1)))

        # Write tracks_to_predict info (indices only)
        f.write(struct.pack("i", _to_int32(len(tracks_to_predict))))
        for track in tracks_to_predict:
            track_index = track.get("track_index", -1)
            f.write(struct.pack("i", _to_int32(track_index, -1)))

        # Count total entities
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", _to_int32(num_objects)))
        f.write(struct.pack("i", _to_int32(num_roads)))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write unique map id
            f.write(struct.pack("i", _to_int32(unique_map_id)))

            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", _to_int32(obj_type)))  # type
            obj_id = obj.get("id", 0)
            f.write(struct.pack("i", _to_int32(obj_id)))  # id
            f.write(struct.pack("i", _to_int32(trajectory_length)))  # array_size
            # Write position arrays
            positions = obj.get("position", [])
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("x", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("y", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("z", 0.0))))

            # Write velocity arrays
            velocities = obj.get("velocity", [])
            for arr, key in [(velocities, "x"), (velocities, "y"), (velocities, "z")]:
                for i in range(trajectory_length):
                    vel = arr[i] if i < len(arr) else {"x": 0.0, "y": 0.0, "z": 0.0}
                    f.write(struct.pack("f", float(vel.get(key, 0.0))))

            # Write heading and valid arrays
            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}f",
                    *[float(headings[i]) if i < len(headings) else 0.0 for i in range(trajectory_length)],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}i",
                    *[_to_int32(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)],
                )
            )

            # Write scalar fields
            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", _to_int32(obj.get("mark_as_expert", 0))))

        # Write roads
        for idx, road in enumerate(map_data.get("roads", [])):
            f.write(struct.pack("i", _to_int32(unique_map_id)))

            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15
            # breakpoint()
            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1, 250)
            size = len(geometry)
            # breakpoint()
            if road_type >= 0 and road_type <= 3:
                road_type = 4
            elif road_type >= 5 and road_type <= 13:
                road_type = 5
            elif road_type >= 14 and road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10
            # Write base entity data
            f.write(struct.pack("i", _to_int32(road_type)))  # type
            road_id = road.get("id", 0)
            f.write(struct.pack("i", _to_int32(road_id)))  # id
            f.write(struct.pack("i", _to_int32(size)))  # array_size

            # Write position arrays
            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))

            # Write scalar fields
            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", _to_int32(road.get("mark_as_expert", 0))))


def load_map(map_name, unique_map_id, binary_output=None, trajectory_length=91):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output, unique_map_id, trajectory_length=trajectory_length)


def _process_single_map(args):
    """Worker function to process a single map file"""
    i, map_path, binary_path, trajectory_length = args
    try:
        load_map(str(map_path), i, str(binary_path), trajectory_length=trajectory_length)
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder="data/processed/training",
    max_maps=50_000,
    num_workers=None,
    shuffle=False,
    trajectory_length=91,
    output_subdir=None,
):
    """Process all maps and save them as binaries using multiprocessing

    Args:
        data_folder: Path to the folder containing JSON map files
        max_maps: Maximum number of maps to process
        num_workers: Number of parallel workers (defaults to cpu_count())
        shuffle: If True, shuffle the JSON files before assigning map IDs.
                 This ensures that when using num_maps < total, you get
                 a random mix of all source maps instead of alphabetically first ones.
    """
    from pathlib import Path
    import random

    if num_workers is None:
        num_workers = cpu_count()

    # Path to the training data
    data_dir = Path(data_folder)
    dataset_name = output_subdir if output_subdir is not None else data_dir.name

    # Create the binaries directory if it doesn't exist
    binary_dir = Path(f"resources/drive/binaries/{dataset_name}")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))

    if shuffle:
        json_files = list(json_files)
        random.shuffle(json_files)

    # Prepare arguments for parallel processing
    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_file = f"map_{i:03d}.bin"
        binary_path = binary_dir / binary_file
        tasks.append((i, map_path, binary_path, trajectory_length))

    # Process maps in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(_process_single_map, tasks), total=len(tasks), desc="Processing maps", unit="map")
        )

    # Collect statistics
    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    if failed > 0:
        print(f"\nFailed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(
        num_agents=num_agents,
        num_maps=1,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        init_steps=0,
        scenario_length=91,
    )

    env.reset()

    tick = 0
    actions = np.stack(
        [np.random.randint(0, space.n + 1, (atn_cache, num_agents)) for space in env.single_action_space], axis=-1
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {num_agents * tick / (time.time() - start)}")

    env.close()


if __name__ == "__main__":
    # test_performance()
    # Process the train dataset
    # process_all_maps(data_folder="/data/processed/training")
    process_all_maps(data_folder="/workspace/ADA/data/nuplan-gpudrive/nuplan")
    # Process the validation/test dataset
    # process_all_maps(data_folder="data/processed/validation")
    # # Process the validation_interactive dataset
    # process_all_maps(data_folder="data/processed/validation_interactive")
