#ifndef ENV_CONFIG_H
#define ENV_CONFIG_H

#include <../../inih-r62/ini.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct{
    char* type;
    float reward_offroad_weight_lb;
    float reward_offroad_weight_ub;
    float reward_collision_weight_lb;
    float reward_collision_weight_ub;
    float reward_goal_weight_lb;
    float reward_goal_weight_ub;
    float entropy_weight_lb;
    float entropy_weight_ub;
    float discount_weight_lb;
    float discount_weight_ub;
} conditioning_config;

// Config struct for parsing INI files - contains all environment configuration
typedef struct
{
    int action_type;
    int dynamics_model;
    float reward_vehicle_collision;
    float reward_offroad_collision;
    float reward_goal;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float reward_ade;
    float goal_radius;
    int collision_behavior;
    int offroad_behavior;
    int spawn_immunity_timer;
    float dt;
    int goal_behavior;
    int scenario_length;
    int init_steps;
    int init_mode;
    int control_mode;
    char* condition_mode;
    int max_controlled_agents;
    conditioning_config* conditioning;
    conditioning_config* co_player_conditioning;
    int co_player_enabled;
    int k_scenarios;
} env_init_config;

// INI file parser handler - parses all environment configuration from drive.ini
static int handler(
    void* config,
    const char* section,
    const char* name,
    const char* value
) {
    env_init_config* env_config = (env_init_config*)config;
    #define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

    if (MATCH("env", "action_type")) {
        if (strcmp(value, "\"discrete\"") == 0 ||strcmp(value, "discrete") == 0) {
            env_config->action_type = 0;  // DISCRETE
        } else if (strcmp(value, "\"continuous\"") == 0 || strcmp(value, "continuous") == 0) {
            env_config->action_type = 1;  // CONTINUOUS
        } else {
            printf("Warning: Unknown action_type value '%s', defaulting to DISCRETE\n", value);
            env_config->action_type = 0;  // Default to DISCRETE
        }
    } else if (MATCH("env", "dynamics_model")) {
        if (strcmp(value, "\"classic\"") == 0 || strcmp(value, "classic") == 0) {
            env_config->dynamics_model = 0;  // CLASSIC
        } else if (strcmp(value, "\"jerk\"") == 0 || strcmp(value, "jerk") == 0) {
            env_config->dynamics_model = 1;  // JERK
        } else {
            printf("Warning: Unknown dynamics_model value '%s', defaulting to JERK\n", value);
            env_config->dynamics_model = 1;  // Default to JERK
        }
    } else if (MATCH("env", "goal_behavior")) {
        env_config->goal_behavior = atoi(value);
    } else if (MATCH("env", "reward_vehicle_collision")) {
        env_config->reward_vehicle_collision = atof(value);
    } else if (MATCH("env", "reward_offroad_collision")) {
        env_config->reward_offroad_collision = atof(value);
    } else if (MATCH("env", "reward_goal")) {
        env_config->reward_goal = atof(value);
    } else if (MATCH("env", "reward_goal_post_respawn")) {
        env_config->reward_goal_post_respawn = atof(value);
    } else if (MATCH("env", "reward_vehicle_collision_post_respawn")) {
        env_config->reward_vehicle_collision_post_respawn = atof(value);
    } else if (MATCH("env", "reward_ade")) {
        env_config->reward_ade = atof(value);
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if(MATCH("env", "collision_behavior")){
        env_config->collision_behavior = atoi(value);
    } else if(MATCH("env", "offroad_behavior")){
        env_config->offroad_behavior = atoi(value);
    } else if (MATCH("env", "spawn_immunity_timer")) {
        env_config->spawn_immunity_timer = atoi(value);
    } else if (MATCH("env", "dt")) {
        env_config->dt = atof(value);
    } else if (MATCH("env", "scenario_length")) {
        env_config->scenario_length = atoi(value);
    } else if (MATCH("env", "init_steps")) {
        env_config->init_steps = atoi(value);
    } else if (MATCH("env", "init_mode")) {
        env_config->init_mode = atoi(value);
    } else if (MATCH("env", "control_mode")) {
        env_config->control_mode = atoi(value);
    }else if (MATCH("env", "co_player_enabled")) {
        if (strcmp(value, "True") == 0 || strcmp(value, "true") == 0 || 
            strcmp(value, "1") == 0) {
            env_config->co_player_enabled = 1;
        } else {
            env_config->co_player_enabled = 0;
        }
    }
    else if (MATCH("env", "k_scenarios")) {
        env_config->k_scenarios = atoi(value);
    }
    else if (MATCH("env.conditioning", "type")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        // Remove quotes if present
        if (value[0] == '"') {
            size_t len = strlen(value) - 2;  // -2 for both quotes
            env_config->conditioning->type = (char*)malloc(len + 1);
            strncpy(env_config->conditioning->type, value + 1, len);
            env_config->conditioning->type[len] = '\0';
        } else {
            env_config->conditioning->type = strdup(value);
        }
    }
    else if (MATCH("env.conditioning", "collision_weight_lb")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->reward_collision_weight_lb = atof(value);
    }
    else if (MATCH("env.conditioning", "collision_weight_ub")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->reward_collision_weight_ub = atof(value);
    }
    else if (MATCH("env.conditioning", "offroad_weight_lb")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->reward_offroad_weight_lb = atof(value);
    }
    else if (MATCH("env.conditioning", "offroad_weight_ub")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->reward_offroad_weight_ub = atof(value);
    }
    else if (MATCH("env.conditioning", "goal_weight_lb")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->reward_goal_weight_lb = atof(value);
    }
    else if (MATCH("env.conditioning", "goal_weight_ub")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->reward_goal_weight_ub = atof(value);
    }
    else if (MATCH("env.conditioning", "entropy_weight_lb")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->entropy_weight_lb = atof(value);
    }
    else if (MATCH("env.conditioning", "entropy_weight_ub")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->entropy_weight_ub = atof(value);
    }
    else if (MATCH("env.conditioning", "discount_weight_lb")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->discount_weight_lb = atof(value);
    }
    else if (MATCH("env.conditioning", "discount_weight_ub")) {
        if (env_config->conditioning == NULL) {
            env_config->conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->conditioning->discount_weight_ub = atof(value);
    }
    
    // Add co_player_policy.conditioning section parsing
    else if (MATCH("env.co_player_policy.conditioning", "type")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        if (value[0] == '"') {
            size_t len = strlen(value) - 2;
            env_config->co_player_conditioning->type = (char*)malloc(len + 1);
            strncpy(env_config->co_player_conditioning->type, value + 1, len);
            env_config->co_player_conditioning->type[len] = '\0';
        } else {
            env_config->co_player_conditioning->type = strdup(value);
        }
    }
    else if (MATCH("env.co_player_policy.conditioning", "collision_weight_lb")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->reward_collision_weight_lb = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "collision_weight_ub")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->reward_collision_weight_ub = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "offroad_weight_lb")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->reward_offroad_weight_lb = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "offroad_weight_ub")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->reward_offroad_weight_ub = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "goal_weight_lb")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->reward_goal_weight_lb = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "goal_weight_ub")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->reward_goal_weight_ub = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "entropy_weight_lb")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->entropy_weight_lb = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "entropy_weight_ub")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->entropy_weight_ub = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "discount_weight_lb")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->discount_weight_lb = atof(value);
    }
    else if (MATCH("env.co_player_policy.conditioning", "discount_weight_ub")) {
        if (env_config->co_player_conditioning == NULL) {
            env_config->co_player_conditioning = (conditioning_config*)malloc(sizeof(conditioning_config));
        }
        env_config->co_player_conditioning->discount_weight_ub = atof(value);
    }

    #undef MATCH
    return 1;
}

#endif // ENV_CONFIG_H
