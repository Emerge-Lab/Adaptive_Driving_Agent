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

        # Remove goal_radius_end - it's used by training loop for curriculum, not Drive
        kwargs.pop("goal_radius_end", None)

        kwargs["resample_frequency"] = self.k_scenarios * self.scenario_length
        self.episode_length = kwargs["resample_frequency"]
        # print(f"resample frequency is ", kwargs["resample_frequency"], flush=True)
        super().__init__(**kwargs)
