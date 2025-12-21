import numpy as np
import gymnasium
import json
import struct
import os
import pufferlib
from pufferlib.ocean.drive import binding
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


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
        reward_ade=0.0,
        goal_behavior=0,
        goal_radius=2.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        scenario_length=None,
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
    ):
        # env
        self.dt = dt
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_behavior = goal_behavior
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.reward_ade = reward_ade
        self.human_agent_idx = human_agent_idx
        self.scenario_length = scenario_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.ini_file = ini_file

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
        base_ego_dim = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(dynamics_model)

        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = binding.MAX_AGENTS - 1
        self.partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES
        self.num_obs = (
            base_ego_dim
            + conditioning_dims
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        # Co-player policy setup
        self.population_play = co_player_enabled
        self.num_agents = num_agents
        self.num_ego_agents = num_ego_agents if self.population_play else num_agents

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
                f"control_mode must be one of 'control_vehicles', 'control_wosac', or 'control_agents'. Got: {self.control_mode_str}"
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

        # Check if resources directory exists
        binary_path = f"{map_dir}/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len([name for name in os.listdir(map_dir) if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to resources/drive/binaries."
            )
        if self.population_play:
            if self.num_ego_agents > num_agents:
                raise ValueError(
                    f"num ego agents ({self.num_ego_agents}) exceeds the number of total agents ({num_agents}))"
                )
            if self.condition_type != "none" and self.co_player_condition_type != "none":
                raise NotImplementedError("Only one agent can be conditioned at once")

        self.max_controlled_agents = int(max_controlled_agents)
        self.use_all_maps = use_all_maps

        self._set_env_variables()

        if self.population_play:
            self.co_player_policy_name = co_player_policy.get("policy_name")
            self.co_player_rnn_name = co_player_policy.get("rnn_name")
            self.co_player_policy = co_player_policy.get("co_player_policy_func")
            self._set_co_player_state()

        super().__init__(buf=buf)
        if self.population_play:
            self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_ego_agents)
            co_player_atn_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_co_players)
            if isinstance(self.single_action_space, pufferlib.spaces.Box):
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=co_player_atn_space.dtype)
            else:
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=np.int32)

        env_ids = []
        for i in range(self.num_envs):
            cur = self.agent_offsets[i]
            nxt = self.agent_offsets[i + 1]
            env_id = binding.env_init(
                self.observations[cur:nxt],
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
                reward_ade=reward_ade,
                goal_radius=goal_radius,
                goal_behavior=self.goal_behavior,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                scenario_length=(int(scenario_length) if scenario_length is not None else None),
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
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        info = []
        if self.population_play:
            info.append(self.ego_ids)
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
            use_all_maps=self.use_all_maps,
            population_play=self.population_play,
            num_ego_agents=self.num_ego_agents,
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
            # When use_all_maps is True, num_agents should be updated from agent_offsets
            if self.use_all_maps:
                self.num_agents = self.agent_offsets[-1]
            self.ego_ids = [i for i in range(self.agent_offsets[-1])]
            if len(self.ego_ids) != self.num_agents:
                raise ValueError("mismatch between number of ego agents and number of agents")
            self.local_co_player_ids = [[] for i in range(self.num_envs)]
            self.local_ego_ids = [[0] for i in range(self.num_envs)]

    def get_co_player_actions(self):
        with torch.no_grad():
            co_player_obs = self.observations[self.co_player_ids]
            # Add conditioning to co-player observations if needed
            if self.co_player_condition_type != "none":
                co_player_obs = self._add_co_player_conditioning(co_player_obs)

            co_player_obs = torch.as_tensor(co_player_obs)
            logits, value = self.co_player_policy.forward_eval(co_player_obs, self.state)
            co_player_action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            co_player_action = co_player_action.cpu().numpy().reshape(self.co_player_actions.shape)
        return co_player_action

    def _set_co_player_state(self):
        with torch.no_grad():
            self.state = dict(
                lstm_h=torch.zeros(self.num_co_players, self.co_player_policy.hidden_size),
                lstm_c=torch.zeros(self.num_co_players, self.co_player_policy.hidden_size),
            )

    def _reset_co_player_state(self, done_indices=None):
        """Reset LSTM state for co-players whose episodes ended"""
        with torch.no_grad():
            if done_indices is None:
                # Reset all
                self._set_co_player_state()
            else:
                # Reset only specific co-players
                device = self.state["lstm_h"].device
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

        return np.concatenate([observations[:, :7], self.cached_conditioning_array, observations[:, 7:]], axis=1)

    def _set_co_player_conditioning(self):
        """Sample and store conditioning values for each environment and update all caches"""
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

        # Add a count of how many agents this represents
        if "n" in last_metrics:
            delta_metrics["ada_agent_count"] = last_metrics["n"]

        return delta_metrics

    def step(self, actions):
        self.terminals[:] = 0

        self.actions[self.ego_ids] = actions

        if self.population_play:
            co_player_actions = self.get_co_player_actions()
            self.actions[self.co_player_ids] = co_player_actions

        binding.vec_step(self.c_envs)

        self.tick += 1
        info = []

        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                if self.adaptive_driving_agent:
                    self.current_scenario_infos.append(log)

                    # Only append to info if we're in the 0th scenario
                    if self.current_scenario == 0:
                        info.append(log)
                        print("0th scenario metrics are ", log, flush=True)
                else:
                    # Non-adaptive mode: always append
                    info.append(log)
                    print("Regular metrics are ", log, flush=True)

        if self.tick % self.scenario_length == 0:
            if self.adaptive_driving_agent and self.current_scenario_infos:
                scenario_log = self._aggregate_scenario_metrics(self.current_scenario_infos)
                scenario_log["scenario_id"] = self.current_scenario
                self.scenario_metrics.append(scenario_log)

                if self.current_scenario == self.k_scenarios - 1:
                    delta_metrics = self._compute_delta_metrics()
                    if delta_metrics:
                        info.append(delta_metrics)
                        print("delta metrics are ", delta_metrics, flush=True)

                    self.scenario_metrics = []

                self.current_scenario_infos = []

            self.current_scenario = (self.current_scenario + 1) % self.k_scenarios

        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            will_resample = 1
            if will_resample:
                # Log deltas before resampling if we're at the end of a cycle
                if self.adaptive_driving_agent and self.scenario_metrics:
                    delta_metrics = self._compute_delta_metrics()
                    if delta_metrics:
                        info.append(delta_metrics)
                        print("delta metrics 2, are ", delta_metrics, flush=True)
                    self.scenario_metrics = []
                    self.current_scenario_infos = []
                    self.current_scenario = 0

                binding.vec_close(self.c_envs)
                self._set_env_variables()
                env_ids = []
                seed = np.random.randint(0, 2**32 - 1)
                for i in range(self.num_envs):
                    cur = self.agent_offsets[i]
                    nxt = self.agent_offsets[i + 1]
                    env_id = binding.env_init(
                        self.observations[cur:nxt],
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
                        reward_goal=self.reward_goal,
                        reward_goal_post_respawn=self.reward_goal_post_respawn,
                        reward_ade=self.reward_ade,
                        goal_radius=self.goal_radius,
                        goal_behavior=self.goal_behavior,
                        collision_behavior=self.collision_behavior,
                        offroad_behavior=self.offroad_behavior,
                        dt=self.dt,
                        scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                        termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
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
                    )
                    env_ids.append(env_id)
                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs, seed)
                self.terminals[:] = 1

        if self.population_play:
            info.append(self.ego_ids)

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

    def render(self):
        binding.vec_render(self.c_envs, 0)

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


def save_map_binary(map_data, output_file, unique_map_id):
    trajectory_length = 91
    """Saves map data in a binary format readable by C"""
    with open(output_file, "wb") as f:
        # Get metadata
        metadata = map_data.get("metadata", {})
        sdc_track_index = metadata.get("sdc_track_index", -1)  # -1 as default if not found
        tracks_to_predict = metadata.get("tracks_to_predict", [])

        # Write sdc_track_index
        f.write(struct.pack("i", sdc_track_index))

        # Write tracks_to_predict info (indices only)
        f.write(struct.pack("i", len(tracks_to_predict)))
        for track in tracks_to_predict:
            track_index = track.get("track_index", -1)
            f.write(struct.pack("i", track_index))

        # Count total entities
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write unique map id
            f.write(struct.pack("i", unique_map_id))

            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))  # type
            f.write(struct.pack("i", obj.get("id", 0)))  # id
            f.write(struct.pack("i", trajectory_length))  # array_size
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
                    *[int(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)],
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
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for idx, road in enumerate(map_data.get("roads", [])):
            f.write(struct.pack("i", unique_map_id))

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
            f.write(struct.pack("i", road_type))  # type
            f.write(struct.pack("i", road.get("id", 0)))  # id
            f.write(struct.pack("i", size))  # array_size

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
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))


def load_map(map_name, unique_map_id, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output, unique_map_id)


def _process_single_map(args):
    """Worker function to process a single map file"""
    i, map_path, binary_path = args
    try:
        load_map(str(map_path), i, str(binary_path))
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder="data/processed/training",
    max_maps=10_000,
    num_workers=None,
):
    """Process all maps and save them as binaries using multiprocessing

    Args:
        data_folder: Path to the folder containing JSON map files
        max_maps: Maximum number of maps to process
        num_workers: Number of parallel workers (defaults to cpu_count())
    """
    from pathlib import Path

    if num_workers is None:
        num_workers = cpu_count()

    # Path to the training data
    data_dir = Path(data_folder)
    dataset_name = data_dir.name

    # Create the binaries directory if it doesn't exist
    binary_dir = Path(f"resources/drive/binaries/{dataset_name}")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))

    # Prepare arguments for parallel processing
    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_file = f"map_{i:03d}.bin"
        binary_path = binary_dir / binary_file
        tasks.append((i, map_path, binary_path))

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
    process_all_maps(data_folder="data/processed/training")
    # Process the validation/test dataset
    # process_all_maps(data_folder="data/processed/validation")
    # # Process the validation_interactive dataset
    # process_all_maps(data_folder="data/processed/validation_interactive")
