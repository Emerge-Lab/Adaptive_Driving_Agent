import numpy as np
import gymnasium
import json
import struct
import os
import pufferlib
from pufferlib.ocean.drive import binding
import torch
import sys
import psutil

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
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        condition_type="none",
        collision_weight_lb=-0.5,
        collision_weight_ub=-0.5,
        offroad_weight_lb=-0.2,
        offroad_weight_ub=-0.2,
        goal_weight_lb=1.0,
        goal_weight_ub=1.0,
        entropy_weight_lb=0.001,
        entropy_weight_ub=0.001,
        discount_weight_lb=0.98,
        discount_weight_ub=0.98,
        max_controlled_agents=-1,
        buf=None,
        ini_file="pufferlib/config/ocean/drive.ini",
        seed=1,
        population_play=False,
        co_player_policy_path=None,
        co_player_policy_name=None,
        co_player_policy=None,
        co_player_rnn_name=None,
        co_player_rnn=None,
        co_player_condition_type=None,
        num_ego_agents=512,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        k_scenarios=0,
        adaptive_driving_agent=False,
<<<<<<< Updated upstream
=======
        ini_file=None,
        # Population play parameters
        population_play=False,
        num_ego_agents=512,
        ego_probability=0.9,
        co_player_policy_name=None,
        co_player_rnn_name=None,
        co_player_policy_path=None,
        co_player_policy=None,
        co_player_rnn=None,
        co_player_condition_type="none",
        co_player_collision_weight_lb=-0.5,
        co_player_collision_weight_ub=-0.5,
        co_player_offroad_weight_lb=-0.2,
        co_player_offroad_weight_ub=-0.2,
        co_player_goal_weight_lb=1.0,
        co_player_goal_weight_ub=1.0,
        co_player_entropy_weight_lb=0.001,
        co_player_entropy_weight_ub=0.001,
        co_player_discount_weight_lb=0.98,
        co_player_discount_weight_ub=0.98,
>>>>>>> Stashed changes
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
        self.resample_frequency = resample_frequency
        self.ini_file = ini_file

<<<<<<< Updated upstream
=======
        # Adaptive driving agent setup
        self.adaptive_driving_agent = int(adaptive_driving_agent)
        self.k_scenarios = int(k_scenarios)

        # Population play setup
        self.population_play = population_play
        self.num_ego_agents = num_ego_agents
        self.ego_probability = ego_probability
        self.co_player_policy_name = co_player_policy_name
        self.co_player_rnn_name = co_player_rnn_name
        self.co_player_policy_path = co_player_policy_path
        self.co_player_policy = co_player_policy
        self.co_player_rnn = co_player_rnn

        # Co-player conditioning setup
        self.co_player_condition_type = co_player_condition_type
        self.co_player_reward_conditioned = co_player_condition_type in ("reward", "all")
        self.co_player_entropy_conditioned = co_player_condition_type in ("entropy", "all")
        self.co_player_discount_conditioned = co_player_condition_type in ("discount", "all")
        self.co_player_collision_weight_lb = co_player_collision_weight_lb
        self.co_player_collision_weight_ub = co_player_collision_weight_ub
        self.co_player_offroad_weight_lb = co_player_offroad_weight_lb
        self.co_player_offroad_weight_ub = co_player_offroad_weight_ub
        self.co_player_goal_weight_lb = co_player_goal_weight_lb
        self.co_player_goal_weight_ub = co_player_goal_weight_ub
        self.co_player_entropy_weight_lb = co_player_entropy_weight_lb
        self.co_player_entropy_weight_ub = co_player_entropy_weight_ub
        self.co_player_discount_weight_lb = co_player_discount_weight_lb
        self.co_player_discount_weight_ub = co_player_discount_weight_ub

>>>>>>> Stashed changes
        # Conditioning setup
        self.condition_type = condition_type
        self.reward_conditioned = condition_type in ("reward", "all")
        self.entropy_conditioned = condition_type in ("entropy", "all")
        self.discount_conditioned = condition_type in ("discount", "all")

        self.collision_weight_lb = collision_weight_lb if self.reward_conditioned else reward_vehicle_collision
        self.collision_weight_ub = collision_weight_ub if self.reward_conditioned else reward_vehicle_collision
        self.offroad_weight_lb = offroad_weight_lb if self.reward_conditioned else reward_offroad_collision
        self.offroad_weight_ub = offroad_weight_ub if self.reward_conditioned else reward_offroad_collision
        self.goal_weight_lb = goal_weight_lb if self.reward_conditioned else 1.0
        self.goal_weight_ub = goal_weight_ub if self.reward_conditioned else 1.0
        self.entropy_weight_lb = entropy_weight_lb
        self.entropy_weight_ub = entropy_weight_ub
        self.discount_weight_lb = discount_weight_lb
        self.discount_weight_ub = discount_weight_ub

        conditioning_dims = (
            (3 if self.reward_conditioned else 0)
            + (1 if self.entropy_conditioned else 0)
            + (1 if self.discount_conditioned else 0)
        )
        self.dynamics_model = dynamics_model

        # Observation space calculation
        base_ego_dim = 10 if self.dynamics_model == "jerk" else 7

        partner_features = 7
        road_features = 7
        max_partner_objects = 63
        max_road_objects = 200
        self.num_obs = (
            base_ego_dim + conditioning_dims + max_partner_objects * partner_features + max_road_objects * road_features
        )
        self.num_ego_agents = num_ego_agents
        

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)
        self.population_play = population_play
        self.num_agents_const = num_agents

        # Action space setup
        self.num_ego_agents = num_ego_agents

        self.num_agents_const = num_agents
        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_tracks_to_predict":
            self.control_mode = 2
        else:
            raise ValueError(
                f"init_mode must be one of 'control_vehicles', 'control_tracks_to_predict', or 'control_agents'. Got: {self.init_mode_str}"
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
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4, 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check resources
        binary_path = "resources/drive/binaries/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len([name for name in os.listdir("resources/drive/binaries") if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to resources/drive/binaries."
            )

<<<<<<< Updated upstream
        if population_play:
            if num_ego_agents > num_agents:
                raise ValueError(
                    f"num ego agents ({num_ego_agents}) exceeds the number of total agents ({num_agents}))"
                )

        self.max_controlled_agents = int(max_controlled_agents)
        self.num_ego_agents = num_ego_agents

        self.co_player_condition_type = co_player_condition_type
        self.co_player_reward_conditioned = co_player_condition_type in ("reward", "all")
        self.co_player_entropy_conditioned = co_player_condition_type in ("entropy", "all")
        self.co_player_discount_conditioned = co_player_condition_type in ("discount", "all")

        self.collision_weight_lb = collision_weight_lb if self.reward_conditioned else reward_vehicle_collision
        self.collision_weight_ub = collision_weight_ub if self.reward_conditioned else reward_vehicle_collision
        self.offroad_weight_lb = offroad_weight_lb if self.reward_conditioned else reward_offroad_collision
        self.offroad_weight_ub = offroad_weight_ub if self.reward_conditioned else reward_offroad_collision
        self.goal_weight_lb = goal_weight_lb if self.reward_conditioned else 1.0
        self.goal_weight_ub = goal_weight_ub if self.reward_conditioned else 1.0
        self.entropy_weight_lb = entropy_weight_lb
        self.entropy_weight_ub = entropy_weight_ub
        self.discount_weight_lb = discount_weight_lb
        self.discount_weight_ub = discount_weight_ub

        self.adaptive_driving_agent = int(adaptive_driving_agent)
        self.k_scenarios = int(k_scenarios)

        my_shared_tuple = binding.shared(
            num_agents=num_agents,
            num_maps=num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=init_steps,
            max_controlled_agents=self.max_controlled_agents,
            population_play=population_play,
            num_ego_agents=self.num_ego_agents,
        )

        if self.population_play:
            agent_offsets, map_ids, num_envs, ego_ids, co_player_ids = my_shared_tuple

            self.num_envs = num_envs

            self.ego_ids = [item for sublist in ego_ids for item in sublist]
            self.co_player_ids = [item for sublist in co_player_ids for item in sublist]

            all_agents = set(range(num_agents))
            ego_set = set(self.ego_ids)
            co_player_set = set(self.co_player_ids)

            if ego_set & co_player_set:
                raise ValueError("Overlap between ego ids and co player ids")

            if ego_set | co_player_set != all_agents:
                raise ValueError("Missing agent ids")

            self.num_ego_agents = len(self.ego_ids)
            self.num_co_players = len(self.co_player_ids)

            self.total_agents = self.num_co_players + self.num_ego_agents
            self.num_agents = self.total_agents

            self.co_player_policy_name = co_player_policy_name
            self.co_player_rnn_name = co_player_rnn_name
            self.co_player_policy = co_player_policy
            self.set_co_player_state()

            if self.co_player_condition_type is not None and self.co_player_condition_type != "none":
                self._set_co_player_conditioning()

            # Build per-environment ID lists
            local_ego_ids = []
            for i in range(num_envs):
                if len(ego_ids[i]) > 0:
                    min_id_in_world = min(ego_ids[i] + co_player_ids[i])
                    local_ego_ids.append([eid - min_id_in_world for eid in ego_ids[i]])
                else:
                    min_id_in_world = min(co_player_ids[i])
                    local_ego_ids.append([])

            local_co_player_ids = []
            for i in range(num_envs):
                if len(ego_ids[i]) > 0:
                    min_id_in_world = min(ego_ids[i] + co_player_ids[i])
                else:
                    min_id_in_world = min(co_player_ids[i])
                local_co_player_ids.append([cid - min_id_in_world for cid in co_player_ids[i]])

            self.local_co_player_ids = local_co_player_ids
            self.local_ego_ids = local_ego_ids

        else:
            agent_offsets, map_ids, num_envs = my_shared_tuple
            self.num_agents = self.num_agents_const
            self.ego_ids = [i for i in range(agent_offsets[-1])]
            local_co_player_ids = [[] for i in range(num_envs)]
            local_ego_ids = [[0] for i in range(num_envs)]

=======
        # Iterate through all maps to count total agents that can be initialized for each map
        if self.population_play:
            agent_offsets, map_ids, num_envs, ego_ids, co_player_ids = binding.shared(
                num_agents=num_ego_agents,
                num_maps=num_maps,
                population_play=True,
                ego_probability=ego_probability,
            )
            # Flatten the ego/co-player ID lists
            self.ego_ids = [item for sublist in ego_ids for item in sublist]
            self.co_player_ids = [item for sublist in co_player_ids for item in sublist]
            self.total_agents = agent_offsets[num_envs]
            self.num_ego_agents_count = len(self.ego_ids)  # Number of ego agents
            self.num_agents = self.total_agents  # Allocate buffers for all agents (ego + co-player)

            # Load co-player policy if path provided
            if self.co_player_policy is None and self.co_player_policy_path:
                import torch
                checkpoint = torch.load(self.co_player_policy_path, map_location='cpu')
                # Policy and RNN will be set by the training loop
                print(f"Loaded co-player checkpoint from {self.co_player_policy_path}")
        else:
            agent_offsets, map_ids, num_envs = binding.shared(
                num_agents=num_agents,
                num_maps=num_maps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                init_steps=init_steps,
                max_controlled_agents=self.max_controlled_agents,
                population_play=False,
            )
            self.num_agents = num_agents
            self.total_agents = num_agents

>>>>>>> Stashed changes
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs

        super().__init__(buf=buf)
<<<<<<< Updated upstream
        if self.population_play:
            self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_ego_agents)
            co_player_atn_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_co_players)
            if isinstance(self.single_action_space, pufferlib.spaces.Box):
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=co_player_atn_space.dtype)
            else:
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=np.int32)

        # Create environments
=======

        # Set up co-player state if population play
        if self.population_play:
            self.set_co_player_state()

>>>>>>> Stashed changes
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]

            # Get ego and co-player IDs for this environment
            if self.population_play:
                env_ego_ids = [agent_id for agent_id in self.ego_ids if cur <= agent_id < nxt]
                env_co_player_ids = [agent_id for agent_id in self.co_player_ids if cur <= agent_id < nxt]
                # Convert to local indices within this environment
                env_ego_ids_local = [agent_id - cur for agent_id in env_ego_ids]
                env_co_player_ids_local = [agent_id - cur for agent_id in env_co_player_ids]
            else:
                env_ego_ids_local = []
                env_co_player_ids_local = []

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
                goal_behavior=goal_behavior,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                scenario_length=(int(scenario_length) if scenario_length is not None else None),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                population_play=self.population_play,
                num_co_players=len(local_co_player_ids[i]),
                co_player_ids=local_co_player_ids[i],
                ego_agent_ids=local_ego_ids[i],
                num_ego_agents=len(local_ego_ids[i]),
                ini_file=self.ini_file,
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                adaptive_driving=self.adaptive_driving_agent,
                k_scenarios=self.k_scenarios,
                # conditioning params
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
<<<<<<< Updated upstream
=======
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                adaptive_driving=self.adaptive_driving_agent,
                k_scenarios=self.k_scenarios,
                # Population play parameters
                population_play=self.population_play,
                num_ego_agents=len(env_ego_ids_local) if self.population_play else 0,
                ego_agent_ids=env_ego_ids_local if self.population_play else [],
                num_co_players=len(env_co_player_ids_local) if self.population_play else 0,
                co_player_ids=env_co_player_ids_local if self.population_play else [],
                # Co-player conditioning parameters
                co_player_use_rc=self.co_player_reward_conditioned,
                co_player_use_ec=self.co_player_entropy_conditioned,
                co_player_use_dc=self.co_player_discount_conditioned,
                co_player_collision_weight_lb=self.co_player_collision_weight_lb,
                co_player_collision_weight_ub=self.co_player_collision_weight_ub,
                co_player_offroad_weight_lb=self.co_player_offroad_weight_lb,
                co_player_offroad_weight_ub=self.co_player_offroad_weight_ub,
                co_player_goal_weight_lb=self.co_player_goal_weight_lb,
                co_player_goal_weight_ub=self.co_player_goal_weight_ub,
                co_player_entropy_weight_lb=self.co_player_entropy_weight_lb,
                co_player_entropy_weight_ub=self.co_player_entropy_weight_ub,
                co_player_discount_weight_lb=self.co_player_discount_weight_lb,
                co_player_discount_weight_ub=self.co_player_discount_weight_ub,
>>>>>>> Stashed changes
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

<<<<<<< Updated upstream
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

    def set_co_player_state(self): ## set in init (state doesnt get updated anywhere else)
        self.state = dict(
            lstm_h=torch.zeros(self.num_co_players, self.co_player_policy.hidden_size),
            lstm_c=torch.zeros(self.num_co_players, self.co_player_policy.hidden_size),
        )

    def reset_co_player_state(self, done_indices=None):
        """Reset LSTM state for co-players whose episodes ended"""
        if done_indices is None:
            # Reset all
            self.set_co_player_state()
        else:
            # Reset only specific co-players
            device = self.state["lstm_h"].device
            self.state["lstm_h"][done_indices] = 0
            self.state["lstm_c"][done_indices] = 0

    def _add_co_player_conditioning(self, observations):
        """Add pre-sampled conditioning variables to co-player observations"""
        if not (
            self.co_player_reward_conditioned
            or self.co_player_entropy_conditioned
            or self.co_player_discount_conditioned
        ):
            return observations

        conditioning_values = []
        for i in range(self.num_envs):
            num_co_players_in_env = len(self.local_co_player_ids[i])
            if num_co_players_in_env == 0:
                continue

            for _ in range(num_co_players_in_env):
                conditioning_values.append(self.env_conditioning[i])

        conditioning_array = np.stack(conditioning_values, axis=0)

        obs_with_conditioning = np.concatenate(
            [
                observations[:, :7],  # First 7 base observations
                conditioning_array,  # Conditioning variables
                observations[:, 7:],  # Rest of observations
            ],
            axis=1,
        )

        return obs_with_conditioning

    def _set_co_player_conditioning(self):
        """Sample and store conditioning values for each environment"""
        self.env_conditioning = []

        for i in range(self.num_envs):
            env_cond = []

            if self.co_player_reward_conditioned:
                collision_weight = np.random.uniform(self.collision_weight_lb, self.collision_weight_ub)
                offroad_weight = np.random.uniform(self.offroad_weight_lb, self.offroad_weight_ub)
                goal_weight = np.random.uniform(self.goal_weight_lb, self.goal_weight_ub)
                env_cond.extend([collision_weight, offroad_weight, goal_weight])

            if self.co_player_entropy_conditioned:
                entropy_weight = np.random.uniform(self.entropy_weight_lb, self.entropy_weight_ub)
                env_cond.append(entropy_weight)

            if self.co_player_discount_conditioned:
                discount_weight = np.random.uniform(self.discount_weight_lb, self.discount_weight_ub)
                env_cond.append(discount_weight)

            self.env_conditioning.append(np.array(env_cond, dtype=np.float32))
=======
    def set_co_player_state(self):
        """Initialize hidden states for co-player policy."""
        import torch

        if not self.population_play:
            return

        num_co_players = len(self.co_player_ids)
        if num_co_players == 0:
            self.co_player_lstm_h = None
            self.co_player_lstm_c = None
            return

        # Initialize hidden states for co-player RNN
        # These will be updated by the co-player policy during inference
        if self.co_player_rnn is not None:
            hidden_size = self.co_player_rnn.hidden_size if hasattr(self.co_player_rnn, 'hidden_size') else 256
            self.co_player_lstm_h = torch.zeros(num_co_players, hidden_size)
            self.co_player_lstm_c = torch.zeros(num_co_players, hidden_size)
        else:
            self.co_player_lstm_h = None
            self.co_player_lstm_c = None

        # Note: Co-player conditioning weights are now sampled in C code (drive.h init function)

    def get_co_player_actions(self):
        """Run inference on co-player policy to get actions for co-player agents."""
        import torch

        if not self.population_play or len(self.co_player_ids) == 0:
            return np.array([])

        if self.co_player_policy is None or self.co_player_rnn is None:
            # Return random actions if no co-player policy is loaded
            num_co_players = len(self.co_player_ids)
            if self._action_type_flag == 0:  # discrete
                if self.dynamics_model == "classic":
                    return np.random.randint(0, [7, 13], size=(num_co_players, 2))
                else:  # jerk
                    return np.random.randint(0, [4, 3], size=(num_co_players, 2))
            else:  # continuous
                return np.random.uniform(-1, 1, size=(num_co_players, 2)).astype(np.float32)

        # Get observations for co-player agents
        co_player_obs = self.observations[self.co_player_ids]

        with torch.no_grad():
            obs_tensor = torch.from_numpy(co_player_obs).float()

            # Run policy forward pass
            hidden = self.co_player_policy(obs_tensor)

            # Run RNN forward pass
            if self.co_player_lstm_h is not None:
                hidden, (self.co_player_lstm_h, self.co_player_lstm_c) = self.co_player_rnn(
                    hidden, (self.co_player_lstm_h, self.co_player_lstm_c), done=None
                )

            # Get actions from policy
            # Assuming policy has an actor head that produces logits
            if hasattr(self.co_player_policy, 'actor'):
                logits = self.co_player_policy.actor(hidden)
            else:
                logits = hidden

            if self._action_type_flag == 0:  # discrete
                # Sample from categorical distribution
                if self.dynamics_model == "classic":
                    logits_accel = logits[:, :7]
                    logits_steer = logits[:, 7:20]
                    actions_accel = torch.argmax(logits_accel, dim=-1)
                    actions_steer = torch.argmax(logits_steer, dim=-1)
                    actions = torch.stack([actions_accel, actions_steer], dim=-1)
                else:  # jerk
                    logits_long = logits[:, :4]
                    logits_lat = logits[:, 4:7]
                    actions_long = torch.argmax(logits_long, dim=-1)
                    actions_lat = torch.argmax(logits_lat, dim=-1)
                    actions = torch.stack([actions_long, actions_lat], dim=-1)
                return actions.cpu().numpy()
            else:  # continuous
                return torch.tanh(logits).cpu().numpy()
>>>>>>> Stashed changes

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        info = []
        if self.population_play:
            info.append(self.ego_ids)
            self.reset_co_player_state()
        self.tick = 0
<<<<<<< Updated upstream
        return self.observations, info

    def step(self, actions):
        self.terminals[:] = 0
=======
        if self.population_play:
            # Reset co-player hidden states
            self.set_co_player_state()
            # Return only ego agent observations
            return self.observations[self.ego_ids], []
        return self.observations, []

    def step(self, actions):
        self.terminals[:] = 0

        if self.population_play:
            # Set actions for ego agents
            self.actions[self.ego_ids] = actions
            # Get actions for co-player agents
            co_player_actions = self.get_co_player_actions()
            if len(co_player_actions) > 0:
                self.actions[self.co_player_ids] = co_player_actions
        else:
            self.actions[:] = actions

        binding.vec_step(self.c_envs)
>>>>>>> Stashed changes
        self.tick += 1

        self.actions[self.ego_ids] = actions

        if self.population_play:
            co_player_actions = self.get_co_player_actions()
            self.actions[self.co_player_ids] = co_player_actions

        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            will_resample = 1
            if will_resample:
                binding.vec_close(self.c_envs)
<<<<<<< Updated upstream
                my_shared_tuple = binding.shared(
                    num_agents=self.num_agents,
                    num_maps=self.num_maps,
                    init_mode=self.init_mode,
                    control_mode=self.control_mode,
                    init_steps=self.init_steps,
                    max_controlled_agents=self.max_controlled_agents,
                    population_play=self.population_play,
                    num_ego_agents=self.num_ego_agents,
                )
                
                if self.population_play:
                    agent_offsets, map_ids, num_envs, ego_ids, co_player_ids = my_shared_tuple
                    self.ego_ids = [item for sublist in ego_ids for item in sublist]
                    self.co_player_ids = [item for sublist in co_player_ids for item in sublist]
                    self.num_ego_agents = len(self.ego_ids)
                    self.num_co_players = len(self.co_player_ids)
                    self.num_envs = num_envs
                    self.num_agents = self.total_agents = self.num_co_players + self.num_ego_agents
                    self.set_co_player_state()
                    
                    local_co_player_ids = []
                    for i in range(num_envs):
                        env_start = agent_offsets[i]
                        env_end = agent_offsets[i + 1]
                        local_co_player_ids.append(
                            [cid - env_start for cid in co_player_ids[i] if env_start <= cid < env_end]
                        )

                    local_ego_ids = []
                    for i in range(num_envs):
                        env_start = agent_offsets[i]
                        env_end = agent_offsets[i + 1]
                        local_ego_ids.append([eid - env_start for eid in ego_ids[i] if env_start <= eid < env_end])

                    self.local_co_player_ids = local_co_player_ids
                    self.local_ego_ids = local_ego_ids
                    if self.co_player_condition_type is not None and self.co_player_condition_type != "none":
                        self._set_co_player_conditioning()
                else:
                    # Non-population play mode stays the same
                    agent_offsets, map_ids, num_envs = my_shared_tuple
                    self.num_agents = self.num_agents_const
                    self.ego_ids = [i for i in range(agent_offsets[-1])]
                    local_co_player_ids = [[] for i in range(num_envs)]
                    local_ego_ids = [[0] for i in range(num_envs)]  # Single ego agent per env
=======

                # Resample environments
                if self.population_play:
                    agent_offsets, map_ids, num_envs, ego_ids, co_player_ids = binding.shared(
                        num_agents=self.num_ego_agents,
                        num_maps=self.num_maps,
                        population_play=True,
                        ego_probability=self.ego_probability,
                    )
                    # Update ego and co-player IDs
                    self.ego_ids = [item for sublist in ego_ids for item in sublist]
                    self.co_player_ids = [item for sublist in co_player_ids for item in sublist]
                    self.total_agents = agent_offsets[num_envs]
                    self.num_agents = len(self.ego_ids)

                    # Reset co-player state with new number of co-players
                    self.set_co_player_state()
                else:
                    agent_offsets, map_ids, num_envs = binding.shared(
                        num_agents=self.num_agents,
                        num_maps=self.num_maps,
                        init_mode=self.init_mode,
                        control_mode=self.control_mode,
                        init_steps=self.init_steps,
                        max_controlled_agents=self.max_controlled_agents,
                        population_play=False,
                    )
>>>>>>> Stashed changes

                env_ids = []
                seed = np.random.randint(0, 2**32 - 1)
                for i in range(num_envs):
                    cur = agent_offsets[i]
                    nxt = agent_offsets[i + 1]

<<<<<<< Updated upstream
=======
                    # Get ego and co-player IDs for this environment (population play)
                    if self.population_play:
                        env_ego_ids = [agent_id for agent_id in self.ego_ids if cur <= agent_id < nxt]
                        env_co_player_ids = [agent_id for agent_id in self.co_player_ids if cur <= agent_id < nxt]
                        env_ego_ids_local = [agent_id - cur for agent_id in env_ego_ids]
                        env_co_player_ids_local = [agent_id - cur for agent_id in env_co_player_ids]
                    else:
                        env_ego_ids_local = []
                        env_co_player_ids_local = []

>>>>>>> Stashed changes
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
                        max_controlled_agents=self.max_controlled_agents,
                        map_id=map_ids[i],
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
                        init_steps=self.init_steps,
                        init_mode=self.init_mode,
                        control_mode=self.control_mode,
                        population_play=self.population_play,
                        num_co_players=len(local_co_player_ids[i]),
                        co_player_ids=local_co_player_ids[i],
                        ego_agent_ids=local_ego_ids[i],
                        num_ego_agents=len(local_ego_ids[i]),
                        adaptive_driving=self.adaptive_driving_agent,
                        k_scenarios=self.k_scenarios,
                        # Population play parameters
                        population_play=self.population_play,
                        num_ego_agents=len(env_ego_ids_local) if self.population_play else 0,
                        ego_agent_ids=env_ego_ids_local if self.population_play else [],
                        num_co_players=len(env_co_player_ids_local) if self.population_play else 0,
                        co_player_ids=env_co_player_ids_local if self.population_play else [],
                        # Co-player conditioning parameters
                        co_player_use_rc=self.co_player_reward_conditioned,
                        co_player_use_ec=self.co_player_entropy_conditioned,
                        co_player_use_dc=self.co_player_discount_conditioned,
                        co_player_collision_weight_lb=self.co_player_collision_weight_lb,
                        co_player_collision_weight_ub=self.co_player_collision_weight_ub,
                        co_player_offroad_weight_lb=self.co_player_offroad_weight_lb,
                        co_player_offroad_weight_ub=self.co_player_offroad_weight_ub,
                        co_player_goal_weight_lb=self.co_player_goal_weight_lb,
                        co_player_goal_weight_ub=self.co_player_goal_weight_ub,
                        co_player_entropy_weight_lb=self.co_player_entropy_weight_lb,
                        co_player_entropy_weight_ub=self.co_player_entropy_weight_ub,
                        co_player_discount_weight_lb=self.co_player_discount_weight_lb,
                        co_player_discount_weight_ub=self.co_player_discount_weight_ub,
                    )

                    env_ids.append(env_id)

                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs, seed)
                self.terminals[:] = 1
<<<<<<< Updated upstream
        if self.population_play:
            info.append(self.ego_ids)  ## this is used to slice ego and co players correctly later on
=======

        # Return only ego observations/rewards in population play mode
        if self.population_play:
            return (
                self.observations[self.ego_ids],
                self.rewards[self.ego_ids],
                self.terminals[self.ego_ids],
                self.truncations[self.ego_ids],
                info
            )
>>>>>>> Stashed changes
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def get_global_agent_state(self):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs, states["x"], states["y"], states["z"], states["heading"], states["id"]
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

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def simplify_polyline(geometry, polyline_reduction_threshold):
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

            if area < polyline_reduction_threshold:
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
        print(len(map_data.get("objects", [])))
        print(len(map_data.get("roads", [])))
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
                geometry = simplify_polyline(geometry, 0.1)
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


def process_all_maps():
    """Process all maps and save them as binaries"""
    from pathlib import Path

    # Create the binaries directory if it doesn't exist
    binary_dir = Path("resources/drive/binaries")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Path to the training data
    data_dir = Path("")

    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    # Process each JSON file
    for i, map_path in enumerate(json_files[:10000]):
        binary_file = f"map_{i:03d}.bin"  # Use zero-padded numbers for consistent sorting
        binary_path = binary_dir / binary_file

        print(f"Processing {map_path.name} -> {binary_file}")
        # try:
        load_map(str(map_path), i, str(binary_path))
        # except Exception as e:
        #     print(f"Error processing {map_path.name}: {e}")


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
    process_all_maps()
