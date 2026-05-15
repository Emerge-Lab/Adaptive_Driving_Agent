#define Env Drive
#define MY_SHARED
#define MY_PUT

#include <Python.h>
#include "binding.h"

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject *rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);
    // env->truncations is wired from positional args by env_binding.h's
    // env_init handler (zero-copy view of the PufferLib SHM buffer).

    // trial_ended_this_step is OPTIONAL — older callers may not pass it.
    // Defaults to NULL; c_step's memset is guarded.
    PyObject *trial = PyDict_GetItemString(kwargs, "trial_ended_this_step");
    if (trial != NULL) {
        if (!PyObject_TypeCheck(trial, &PyArray_Type)) {
            PyErr_SetString(PyExc_TypeError, "trial_ended_this_step must be a NumPy array");
            return 1;
        }
        PyArrayObject *trial_arr = (PyArrayObject *)trial;
        if (!PyArray_ISCONTIGUOUS(trial_arr) || PyArray_NDIM(trial_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "trial_ended_this_step must be 1D contiguous");
            return 1;
        }
        env->trial_ended_this_step = PyArray_DATA(trial_arr);
    }
    // removed (per-agent off-map flag, B''). Same pattern as
    // trial_ended_this_step: C is the only writer; Python reads.
    PyObject *removed_obj = PyDict_GetItemString(kwargs, "removed");
    if (removed_obj != NULL) {
        if (!PyObject_TypeCheck(removed_obj, &PyArray_Type)) {
            PyErr_SetString(PyExc_TypeError, "removed must be a NumPy array");
            return 1;
        }
        PyArrayObject *removed_arr = (PyArrayObject *)removed_obj;
        if (!PyArray_ISCONTIGUOUS(removed_arr) || PyArray_NDIM(removed_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "removed must be 1D contiguous");
            return 1;
        }
        env->removed = PyArray_DATA(removed_arr);
    }
    return 0;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {

    int population_play = unpack(kwargs, "population_play");
    if (population_play) {
        return my_shared_population_play(self, args, kwargs);
    } else {
        return my_shared_self_play(self, args, kwargs);
    }
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "scenario_length")) {
        conf.scenario_length = (int)unpack(kwargs, "scenario_length");
    }
    if (conf.scenario_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "scenario_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    if (PyDict_GetItemString(kwargs, "dynamics_model")) {
        char *dynamics_str = unpack_str(kwargs, "dynamics_model");
        env->dynamics_model = (strcmp(dynamics_str, "jerk") == 0) ? JERK : CLASSIC;
    }
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->reward_lane_align = (float)unpack(kwargs, "reward_lane_align");
    env->reward_vel_align = (float)unpack(kwargs, "reward_vel_align");
    env->scenario_length = conf.scenario_length;

    // GOAL_TRIAL config (only used when goal_behavior == GOAL_TRIAL).
    env->max_trials_per_episode = 2;
    env->per_trial_timeout = conf.scenario_length;
    if (kwargs && PyDict_GetItemString(kwargs, "max_trials_per_episode")) {
        env->max_trials_per_episode = (int)unpack(kwargs, "max_trials_per_episode");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "per_trial_timeout")) {
        int v = (int)unpack(kwargs, "per_trial_timeout");
        if (v > 0) env->per_trial_timeout = v;  // 0 means "use default" (scenario_length)
    }

    env->termination_mode = conf.termination_mode;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->dt = conf.dt;

    // Conditioning parameters
    env->use_rc = (bool)unpack(kwargs, "use_rc");
    env->use_ec = (bool)unpack(kwargs, "use_ec");
    env->use_dc = (bool)unpack(kwargs, "use_dc");
    env->collision_weight_lb = (float)unpack(kwargs, "collision_weight_lb");
    env->collision_weight_ub = (float)unpack(kwargs, "collision_weight_ub");
    env->offroad_weight_lb = (float)unpack(kwargs, "offroad_weight_lb");
    env->offroad_weight_ub = (float)unpack(kwargs, "offroad_weight_ub");
    env->goal_weight_lb = (float)unpack(kwargs, "goal_weight_lb");
    env->goal_weight_ub = (float)unpack(kwargs, "goal_weight_ub");
    env->entropy_weight_lb = (float)unpack(kwargs, "entropy_weight_lb");
    env->entropy_weight_ub = (float)unpack(kwargs, "entropy_weight_ub");
    env->discount_weight_lb = (float)unpack(kwargs, "discount_weight_lb");
    env->discount_weight_ub = (float)unpack(kwargs, "discount_weight_ub");
    env->population_play = unpack(kwargs, "population_play");

    if (env->population_play) {
        env->num_co_players = unpack(kwargs, "num_co_players");
        double *co_player_ids_d = unpack_float_array(kwargs, "co_player_ids", &env->num_co_players);

        if (co_player_ids_d != NULL && env->num_co_players > 0) {
            env->co_player_ids = (int *)malloc(env->num_co_players * sizeof(int));
            if (env->co_player_ids == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for co_player_ids\n");
                free(co_player_ids_d);
                env->num_co_players = 0;
            } else {
                for (int i = 0; i < env->num_co_players; i++) {
                    env->co_player_ids[i] = (int)co_player_ids_d[i];
                }
                free(co_player_ids_d);
            }
        } else {
            if (co_player_ids_d != NULL) {
                free(co_player_ids_d);
            }
            env->co_player_ids = NULL;
            env->num_co_players = 0;
        }

        // Handle ego agents - always as an array
        env->num_ego_agents = unpack(kwargs, "num_ego_agents");
        if (env->num_ego_agents > 0) {
            double *ego_agent_ids_d = unpack_float_array(kwargs, "ego_agent_ids", &env->num_ego_agents);
            if (ego_agent_ids_d != NULL) {
                env->ego_agent_ids = (int *)malloc(env->num_ego_agents * sizeof(int));
                for (int i = 0; i < env->num_ego_agents; i++) {
                    env->ego_agent_ids[i] = (int)ego_agent_ids_d[i];
                }
                free(ego_agent_ids_d);
            } else {
                env->ego_agent_ids = NULL;
                env->num_ego_agents = 0;
            }
        } else {
            env->ego_agent_ids = NULL;
            env->num_ego_agents = 0;
        }
    } else {
        // Non-population play mode - set defaults
        env->num_ego_agents = 0;
        env->ego_agent_ids = NULL;
    }

    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    // Render mode: 0=RENDER_OFF, 1=RENDER_HEADLESS, 2=RENDER_WINDOW
    env->render_mode = RENDER_OFF; // Default to off
    if (kwargs && PyDict_GetItemString(kwargs, "render_mode")) {
        env->render_mode = (int)unpack(kwargs, "render_mode");
    }
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    char *map_dir = unpack_str(kwargs, "map_dir");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[512];
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;

    // trial_ended_this_step is OPTIONAL. NULL is safe (c_step's memset is guarded).
    env->trial_ended_this_step = NULL;
    PyObject *trial = PyDict_GetItemString(kwargs, "trial_ended_this_step");
    if (trial != NULL) {
        if (!PyObject_TypeCheck(trial, &PyArray_Type)) {
            PyErr_SetString(PyExc_TypeError, "trial_ended_this_step must be a NumPy array");
            return -1;
        }
        PyArrayObject *trial_arr = (PyArrayObject *)trial;
        if (!PyArray_ISCONTIGUOUS(trial_arr) || PyArray_NDIM(trial_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "trial_ended_this_step must be 1D contiguous");
            return -1;
        }
        env->trial_ended_this_step = PyArray_DATA(trial_arr);
    }
    env->removed = NULL;
    PyObject *removed_obj = PyDict_GetItemString(kwargs, "removed");
    if (removed_obj != NULL) {
        if (!PyObject_TypeCheck(removed_obj, &PyArray_Type)) {
            PyErr_SetString(PyExc_TypeError, "removed must be a NumPy array");
            return -1;
        }
        PyArrayObject *removed_arr = (PyArrayObject *)removed_obj;
        if (!PyArray_ISCONTIGUOUS(removed_arr) || PyArray_NDIM(removed_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "removed must be 1D contiguous");
            return -1;
        }
        env->removed = PyArray_DATA(removed_arr);
    }

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "offroad_per_agent", log->offroad_per_agent);
    assign_to_dict(dict, "collisions_per_agent", log->collisions_per_agent);
    assign_to_dict(dict, "goals_sampled_this_episode", log->goals_sampled_this_episode);
    assign_to_dict(dict, "goals_reached_this_episode", log->goals_reached_this_episode);
    assign_to_dict(dict, "speed_at_goal", log->speed_at_goal);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);

    // GOAL_TRIAL metrics (zero under other goal_behavior).
    assign_to_dict(dict, "n_trials_completed", log->n_trials_completed);
    assign_to_dict(dict, "n_trials_goal_reached", log->n_trials_goal_reached);
    assign_to_dict(dict, "n_trials_timed_out", log->n_trials_timed_out);
    if (log->n_trials_completed > 0.0f) {
        assign_to_dict(dict, "trial_mean_length", log->trial_total_length / log->n_trials_completed);
        assign_to_dict(dict, "trial_goal_reach_rate", log->n_trials_goal_reached / log->n_trials_completed);
    } else {
        assign_to_dict(dict, "trial_mean_length", 0.0f);
        assign_to_dict(dict, "trial_goal_reach_rate", 0.0f);
    }
    // Per-trial-index success rate (GOAL_TRIAL only). n_trials_completed is
    // the gate: it's only non-zero under GOAL_TRIAL, so gb=0/1/2 won't leak
    // these keys into wandb / eval output.
    if (log->n_trials_completed > 0.0f) {
        char key[32];
        for (int k = 0; k < N_TRIAL_K_SLOTS; k++) {
            snprintf(key, sizeof(key), "trial_%d_score", k);
            assign_to_dict(dict, key, log->trial_k_goal_reached[k]);
        }
    }
    return 0;
}
