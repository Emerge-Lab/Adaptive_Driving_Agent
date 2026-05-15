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

        # Under GOAL_TRIAL (=3), the user's mental model is "k_scenarios == number
        # of trials per episode" and "trial_length == scenario_length." Force the
        # link: the INI defaults (max_trials_per_episode=2, per_trial_timeout=0)
        # are values you would never want under k≠2 anyway, so we always overwrite
        # under goal_behavior=3. To override per-trial timeout in a launcher, set
        # `--env.per-trial-timeout` to any value > 0; to override max_trials, set
        # `--env.max-trials-per-episode` to a value != k_scenarios (we treat
        # equal-to-k_scenarios as "user wasn't overriding"). Documented in
        # tests/test_gae_decoupling_integration.py.
        if int(kwargs.get("goal_behavior", 0)) == 3:
            # Force max_trials = k_scenarios unless user explicitly passed a
            # value that is neither the INI default (2) nor equal to k_scenarios.
            ini_default = 2
            user_max_trials = int(kwargs.get("max_trials_per_episode", ini_default))
            if user_max_trials == ini_default or user_max_trials == self.k_scenarios:
                kwargs["max_trials_per_episode"] = self.k_scenarios
            # else: user passed something deliberate (e.g. max_trials=5 with
            # k_scenarios=3 for "extra retries"); respect it.
            # per_trial_timeout: INI default is 0 ("use scenario_length in C").
            # Force it to scenario_length so the Python and C buffer budgets
            # match (episode_length = k_scenarios * scenario_length).
            if not kwargs.get("per_trial_timeout"):
                kwargs["per_trial_timeout"] = self.scenario_length

        super().__init__(**kwargs)
