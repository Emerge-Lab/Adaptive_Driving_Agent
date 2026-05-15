from pufferlib.ocean.drive import Drive
import pufferlib


class AdaptiveDrivingAgent(Drive):
    def __init__(self, **kwargs):
        self.env_name = "adaptive_drive"
        self.k_scenarios = kwargs["k_scenarios"]
        self.scenario_length = kwargs["scenario_length"]
        self.dynamics_model = kwargs["dynamics_model"]

        kwargs["ini_file"] = "pufferlib/config/ocean/adaptive.ini"
        kwargs["adaptive_driving_agent"] = True

        # Human replay mode: disable co-players, use human trajectories for other agents
        human_replay_mode = kwargs.pop("human_replay_mode", False)
        if human_replay_mode:
            kwargs["co_player_enabled"] = False

        kwargs["resample_frequency"] = self.k_scenarios * self.scenario_length
        self.episode_length = kwargs["resample_frequency"]

        # Under GOAL_TRIAL: k_scenarios IS the trial count, scenario_length IS
        # per-trial-timeout. No fallback to INI defaults. Tests that need a
        # custom trial budget should override k_scenarios + scenario_length
        # directly.
        if int(kwargs.get("goal_behavior", 0)) == 3:
            kwargs["max_trials_per_episode"] = self.k_scenarios
            kwargs["per_trial_timeout"] = self.scenario_length

        super().__init__(**kwargs)
