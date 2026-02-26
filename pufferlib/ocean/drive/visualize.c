#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <math.h>
#include <raylib.h>
#include "rlgl.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "error.h"
#include "drivenet.h"
#include "libgen.h"
#include "../env_config.h"
#define TRAJECTORY_LENGTH_DEFAULT 91

typedef struct {
    int pipefd[2];
    pid_t pid;
} VideoRecorder;

bool OpenVideo(VideoRecorder *recorder, const char *output_filename, int width, int height) {
    if (pipe(recorder->pipefd) == -1) {
        fprintf(stderr, "Failed to create pipe\n");
        return false;
    }

    recorder->pid = fork();
    if (recorder->pid == -1) {
        fprintf(stderr, "Failed to fork\n");
        return false;
    }

    char size_str[64];
    snprintf(size_str, sizeof(size_str), "%dx%d", width, height);

    if (recorder->pid == 0) { // Child process: run ffmpeg
        close(recorder->pipefd[1]);
        dup2(recorder->pipefd[0], STDIN_FILENO);
        close(recorder->pipefd[0]);
        // Close all other file descriptors to prevent leaks
        for (int fd = 3; fd < 256; fd++) {
            close(fd);
        }
        execlp("ffmpeg", "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", size_str, "-r", "30", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23", "-loglevel", "error",
               output_filename, NULL);
        TraceLog(LOG_ERROR, "Failed to launch ffmpeg");
        return false;
    }

    close(recorder->pipefd[0]); // Close read end in parent
    return true;
}

void WriteFrame(VideoRecorder *recorder, int width, int height) {
    unsigned char *screen_data = rlReadScreenPixels(width, height);
    write(recorder->pipefd[1], screen_data, width * height * 4 * sizeof(*screen_data));
    RL_FREE(screen_data);
}

void CloseVideo(VideoRecorder *recorder) {
    close(recorder->pipefd[1]);
    waitpid(recorder->pid, NULL, 0);
}

void renderTopDownView(Drive *env, Client *client, float map_width, float map_height, int obs, int lasers,
                       int trajectories, int frame_count, float *path, int show_human_logs, int show_grid,
                       int img_width, int img_height, int zoom_in, int current_scenario, int total_scenarios) {
    BeginDrawing();

    // Calculate map center
    float center_x = (env->grid_map->top_left_x + env->grid_map->bottom_right_x) / 2.0f;
    float center_y = (env->grid_map->top_left_y + env->grid_map->bottom_right_y) / 2.0f;

    // Top-down orthographic camera
    Camera3D camera = {0};

    if (zoom_in) {                                       // Zoom in on part of the map
        camera.position = (Vector3){0.0f, 0.0f, 500.0f}; // above the scene
        camera.target = (Vector3){0.0f, 0.0f, 0.0f};     // look at origin
        camera.fovy = map_height;
    } else { // Show full map - center camera on map
        camera.position = (Vector3){center_x, center_y, 500.0f};
        camera.target = (Vector3){center_x, center_y, 0.0f};
        // Use the larger dimension to ensure full map is visible
        camera.fovy = (map_height > map_width) ? map_height * 1.1f : map_width * 1.1f;
    }

    camera.up = (Vector3){0.0f, -1.0f, 0.0f};
    camera.projection = CAMERA_ORTHOGRAPHIC;

    client->width = img_width;
    client->height = img_height;

    Color road = (Color){35, 35, 37, 255};
    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();

    // Draw human replay trajectories if enabled
    if (show_human_logs) {
        for (int i = 0; i < env->active_agent_count; i++) {
            int idx = env->active_agent_indices[i];
            Vector3 prev_point = {0};
            bool has_prev = false;

            for (int j = 0; j < env->entities[idx].array_size; j++) {
                float x = env->entities[idx].traj_x[j];
                float y = env->entities[idx].traj_y[j];
                float valid = env->entities[idx].traj_valid[j];

                if (!valid) {
                    has_prev = false;
                    continue;
                }

                Vector3 curr_point = {x, y, 0.5f};

                if (has_prev) {
                    DrawLine3D(prev_point, curr_point, Fade(LIGHTGREEN, 0.6f));
                }

                prev_point = curr_point;
                has_prev = true;
            }
        }
    }

    // Draw agent trajs
    if (trajectories) {
        for (int i = 0; i < frame_count; i++) {
            DrawSphere((Vector3){path[i * 2], path[i * 2 + 1], 0.8f}, 0.5f, YELLOW);
        }
    }

    // Draw scene
    draw_scene(env, client, 1, obs, lasers, show_grid);
    EndMode3D();

    // Draw scenario counter overlay (2D text on top of 3D scene)
    char scenario_text[64];
    snprintf(scenario_text, sizeof(scenario_text), "Scenario %d / %d", current_scenario, total_scenarios);
    DrawText(scenario_text, 20, 20, 30, WHITE);

    EndDrawing();
}

void renderAgentView(Drive *env, Client *client, int map_height, int obs_only, int lasers, int show_grid) {
    // Agent perspective camera following the selected agent
    int agent_idx = env->active_agent_indices[env->human_agent_idx];
    Entity *agent = &env->entities[agent_idx];

    BeginDrawing();

    Camera3D camera = {0};
    // Position camera behind and above the agent
    camera.position =
        (Vector3){agent->x - (25.0f * cosf(agent->heading)), agent->y - (25.0f * sinf(agent->heading)), 15.0f};
    camera.target = (Vector3){agent->x + 40.0f * cosf(agent->heading), agent->y + 40.0f * sinf(agent->heading), 1.0f};
    camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    Color road = (Color){35, 35, 37, 255};

    ClearBackground(road);
    BeginMode3D(camera);
    rlEnableDepthTest();
    draw_scene(env, client, 0, obs_only, lasers, show_grid); // mode=0 for agent view
    EndMode3D();
    EndDrawing();
}

static int run_cmd(const char *cmd) {
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "[ffmpeg] command failed (%d): %s\n", rc, cmd);
    }
    return rc;
}

// Make a high-quality GIF from numbered PNG frames like frame_000.png
static int make_gif_from_frames(const char *pattern, int fps, const char *palette_path, const char *out_gif) {
    char cmd[1024];

    // 1) Generate palette (no quotes needed for simple filter)
    //    NOTE: if your frames start at 000, you don't need -start_number.
    snprintf(cmd, sizeof(cmd), "ffmpeg -y -framerate %d -i %s -vf palettegen %s", fps, pattern, palette_path);
    if (run_cmd(cmd) != 0)
        return -1;

    // 2) Use palette to encode the GIF
    snprintf(cmd, sizeof(cmd), "ffmpeg -y -framerate %d -i %s -i %s -lavfi paletteuse -loop 0 %s", fps, pattern,
             palette_path, out_gif);
    if (run_cmd(cmd) != 0)
        return -1;

    return 0;
}

// Transform observations from ego format to co-player format by inserting conditioning values
// src_obs: Source observations (may include ego conditioning)
// dst_obs: Destination buffer for co-player format (with co-player conditioning)
// num_agents: Number of agents to transform
// ego_base_dim: Base ego features (7 for CLASSIC, 10 for JERK)
// co_use_rc/ec/dc: Co-player conditioning flags (determines which features to insert)
void transform_obs_for_coplayer(float *src_obs, float *dst_obs, int num_agents, int ego_obs_size, int coplayer_obs_size,
                                int ego_base_dim, int co_use_rc, int co_use_ec, int co_use_dc, float collision_lb,
                                float collision_ub, float offroad_lb, float offroad_ub, float goal_lb, float goal_ub,
                                float entropy_lb, float entropy_ub, float discount_lb, float discount_ub) {
    // Fixed sizes for partner and road features (from drive.h constants)
    int partner_features = (MAX_AGENTS - 1) * PARTNER_FEATURES;
    int road_features = MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES;
    int partner_road_features = partner_features + road_features;

    // Derive source conditioning size from ego_obs_size
    // This handles cases where ego policy has conditioning (type != "none")
    int src_conditioning = ego_obs_size - ego_base_dim - partner_road_features;

    // Calculate destination conditioning size from flags
    int dst_conditioning = (co_use_rc ? 3 : 0) + (co_use_ec ? 1 : 0) + (co_use_dc ? 1 : 0);

    for (int i = 0; i < num_agents; i++) {
        float *src = src_obs + i * ego_obs_size;
        float *dst = dst_obs + i * coplayer_obs_size;

        // Copy ego base features (without conditioning)
        memcpy(dst, src, ego_base_dim * sizeof(float));

        // Sample and insert conditioning values based on flags
        // Order must match: reward (3), entropy (1), discount (1)
        int cond_idx = ego_base_dim;
        if (co_use_rc) {
            // Reward conditioning (3 features: collision, offroad, goal)
            dst[cond_idx++] = collision_lb + (float)rand() / RAND_MAX * (collision_ub - collision_lb);
            dst[cond_idx++] = offroad_lb + (float)rand() / RAND_MAX * (offroad_ub - offroad_lb);
            dst[cond_idx++] = goal_lb + (float)rand() / RAND_MAX * (goal_ub - goal_lb);
        }
        if (co_use_ec) {
            // Entropy conditioning (1 feature)
            dst[cond_idx++] = entropy_lb + (float)rand() / RAND_MAX * (entropy_ub - entropy_lb);
        }
        if (co_use_dc) {
            // Discount conditioning (1 feature)
            dst[cond_idx++] = discount_lb + (float)rand() / RAND_MAX * (discount_ub - discount_lb);
        }

        // Copy partner + road features, skipping over any source conditioning
        memcpy(dst + ego_base_dim + dst_conditioning, src + ego_base_dim + src_conditioning,
               partner_road_features * sizeof(float));
    }
}

// Helper function for dual-policy forward pass
// Runs ego policy on first num_ego_agents, co-player policy on the rest
// Handles different observation sizes between ego and co-player policies
void forward_population(DriveNet *ego_net, DriveNet *co_player_net, float *observations, int *actions,
                        int num_ego_agents, int num_co_players, int ego_obs_size, int coplayer_obs_size,
                        int ego_base_dim, int co_use_rc, int co_use_ec, int co_use_dc, float co_collision_lb,
                        float co_collision_ub, float co_offroad_lb, float co_offroad_ub, float co_goal_lb,
                        float co_goal_ub, float co_entropy_lb, float co_entropy_ub, float co_discount_lb,
                        float co_discount_ub) {
    if (co_player_net == NULL || num_co_players == 0) {
        // Single policy mode - use ego net for all agents
        forward(ego_net, observations, actions);
        return;
    }

    // Allocate temporary buffers for ego observations/actions
    float *ego_obs = (float *)malloc(num_ego_agents * ego_obs_size * sizeof(float));
    int *ego_actions = (int *)malloc(num_ego_agents * sizeof(int));

    // Allocate temporary buffers for co-player observations/actions
    float *co_obs_raw = observations + num_ego_agents * ego_obs_size;
    float *co_obs_transformed = (float *)malloc(num_co_players * coplayer_obs_size * sizeof(float));
    int *co_actions = (int *)malloc(num_co_players * sizeof(int));

    // Copy ego observations (already correct format)
    memcpy(ego_obs, observations, num_ego_agents * ego_obs_size * sizeof(float));

    // Transform co-player observations (add conditioning features)
    transform_obs_for_coplayer(co_obs_raw, co_obs_transformed, num_co_players, ego_obs_size, coplayer_obs_size,
                               ego_base_dim, co_use_rc, co_use_ec, co_use_dc, co_collision_lb, co_collision_ub,
                               co_offroad_lb, co_offroad_ub, co_goal_lb, co_goal_ub, co_entropy_lb, co_entropy_ub,
                               co_discount_lb, co_discount_ub);

    // Run forward on each network
    forward(ego_net, ego_obs, ego_actions);
    forward(co_player_net, co_obs_transformed, co_actions);

    // Combine actions back
    memcpy(actions, ego_actions, num_ego_agents * sizeof(int));
    memcpy(actions + num_ego_agents, co_actions, num_co_players * sizeof(int));

    // Cleanup
    free(ego_obs);
    free(ego_actions);
    free(co_obs_transformed);
    free(co_actions);
}

int eval_gif(const char *map_name, const char *policy_name, int show_grid, int obs_only, int lasers,
             int show_human_logs, int frame_skip, const char *view_mode, const char *output_topdown,
             const char *output_agent, int num_maps, int zoom_in, const char *ini_file, int k_scenarios_cli,
             int max_controlled_agents_cli, const char *co_player_policy_name) {

    // Parse configuration from INI file
    env_init_config conf = {0};
    if (ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s. Cannot determine environment configuration.\n", ini_file);
        return -1;
    }

    char map_buffer[100];
    if (map_name == NULL) {
        srand(time(NULL));
        int random_map = rand() % num_maps;
        sprintf(map_buffer, "%s/map_%03d.bin", conf.map_dir, random_map);
        map_name = map_buffer;
    }

    if (frame_skip <= 0) {
        frame_skip = 1;
    }

    // Check if map file exists
    FILE *map_file = fopen(map_name, "rb");
    if (map_file == NULL) {
        RAISE_FILE_ERROR(map_name);
    }
    fclose(map_file);

    FILE *policy_file = fopen(policy_name, "rb");
    if (policy_file == NULL) {
        RAISE_FILE_ERROR(policy_name);
    }
    fclose(policy_file);

    int use_rc = (conf.conditioning != NULL)
                     ? (strcmp(conf.conditioning->type, "reward") == 0 || strcmp(conf.conditioning->type, "all") == 0)
                     : 0;
    int use_ec = (conf.conditioning != NULL)
                     ? (strcmp(conf.conditioning->type, "entropy") == 0 || strcmp(conf.conditioning->type, "all") == 0)
                     : 0;
    int use_dc = (conf.conditioning != NULL)
                     ? (strcmp(conf.conditioning->type, "discount") == 0 || strcmp(conf.conditioning->type, "all") == 0)
                     : 0;
    // Initialize environment with all config values from INI [env] section
    Drive env = {
        .action_type = conf.action_type,
        .dynamics_model = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_goal = conf.reward_goal,
        .reward_goal_post_respawn = conf.reward_goal_post_respawn,
        .goal_radius = conf.goal_radius,
        .goal_behavior = conf.goal_behavior,
        .goal_target_distance = conf.goal_target_distance,
        .goal_speed = conf.goal_speed,
        .dt = conf.dt,
        .scenario_length = conf.scenario_length,
        .termination_mode = conf.termination_mode,
        .collision_behavior = conf.collision_behavior,
        .offroad_behavior = conf.offroad_behavior,
        .init_steps = conf.init_steps,
        .init_mode = conf.init_mode,
        .control_mode = conf.control_mode,
        .map_name = (char *)map_name,
        .use_rc = use_rc,
        .use_ec = use_ec,
        .use_dc = use_dc,
        .collision_weight_lb = (conf.conditioning != NULL) ? conf.conditioning->reward_collision_weight_lb : 0.0f,
        .collision_weight_ub = (conf.conditioning != NULL) ? conf.conditioning->reward_collision_weight_ub : 0.0f,
        .offroad_weight_lb = (conf.conditioning != NULL) ? conf.conditioning->reward_offroad_weight_lb : 0.0f,
        .offroad_weight_ub = (conf.conditioning != NULL) ? conf.conditioning->reward_offroad_weight_ub : 0.0f,
        .goal_weight_lb = (conf.conditioning != NULL) ? conf.conditioning->reward_goal_weight_lb : 0.0f,
        .goal_weight_ub = (conf.conditioning != NULL) ? conf.conditioning->reward_goal_weight_ub : 0.0f,
        .entropy_weight_lb = (conf.conditioning != NULL) ? conf.conditioning->entropy_weight_lb : 0.0f,
        .entropy_weight_ub = (conf.conditioning != NULL) ? conf.conditioning->entropy_weight_ub : 0.0f,
        .discount_weight_lb = (conf.conditioning != NULL) ? conf.conditioning->discount_weight_lb : 0.0f,
        .discount_weight_ub = (conf.conditioning != NULL) ? conf.conditioning->discount_weight_ub : 0.0f,
        .max_controlled_agents = (max_controlled_agents_cli > 0) ? max_controlled_agents_cli : 32,
    };

    allocate(&env);

    // Check if map has any active agents
    if (env.active_agent_count == 0) {
        fprintf(stderr, "Error: Map %s has no controllable agents\n", map_name);
        free_allocated(&env);
        return -1;
    }

    // Set which vehicle to focus on for obs mode
    int random_agent_idx = rand() % env.active_agent_count;
    env.human_agent_idx = random_agent_idx;

    c_reset(&env);

    // Make client for rendering
    Client *client = (Client *)calloc(1, sizeof(Client));
    env.client = client;

    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    SetTargetFPS(6000);

    float map_width = env.grid_map->bottom_right_x - env.grid_map->top_left_x;
    float map_height = env.grid_map->top_left_y - env.grid_map->bottom_right_y;

    printf("Map size: %.1fx%.1f\n", map_width, map_height);
    float scale = 6.0f;

    int img_width = (int)roundf(map_width * scale / 2.0f) * 2;
    int img_height = (int)roundf(map_height * scale / 2.0f) * 2;

    InitWindow(img_width, img_height, "Puffer Drive");
    SetConfigFlags(FLAG_MSAA_4X_HINT);

    // Load the textures and models
    client->puffers = LoadTexture("resources/puffers_128.png");
    client->cars[0] = LoadModel("resources/drive/RedCar.glb");
    client->cars[1] = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2] = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3] = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4] = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5] = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");

    // Determine number of ego agents vs co-players
    int num_ego_agents = env.active_agent_count;
    int num_co_players = 0;
    DriveNet *co_player_net = NULL;

    // Co-player conditioning flags (hoisted to outer scope for later use)
    int co_use_rc = 0, co_use_ec = 0, co_use_dc = 0;

    // Check if co-player policy is provided (either via CLI or INI)
    const char *actual_co_player_policy = co_player_policy_name;
    if (actual_co_player_policy == NULL && conf.co_player_enabled && strlen(conf.co_player_policy_path) > 0) {
        actual_co_player_policy = conf.co_player_policy_path;
    }

    if (actual_co_player_policy != NULL) {
        // Population play mode - split agents between ego and co-player
        // Use num_ego_agents from config, or default to half
        if (conf.num_ego_agents > 0 && conf.num_ego_agents < env.active_agent_count) {
            num_ego_agents = conf.num_ego_agents;
        } else {
            num_ego_agents = env.active_agent_count / 2;
        }
        num_co_players = env.active_agent_count - num_ego_agents;

        printf("Population play: %d ego agents, %d co-players\n", num_ego_agents, num_co_players);

        // Load co-player policy
        FILE *co_policy_file = fopen(actual_co_player_policy, "rb");
        if (co_policy_file != NULL) {
            fclose(co_policy_file);
            Weights *co_weights = load_weights(actual_co_player_policy);

            // Determine co-player conditioning from config
            if (conf.co_player_conditioning != NULL) {
                co_use_rc = (strcmp(conf.co_player_conditioning->type, "reward") == 0 ||
                             strcmp(conf.co_player_conditioning->type, "all") == 0);
                co_use_ec = (strcmp(conf.co_player_conditioning->type, "entropy") == 0 ||
                             strcmp(conf.co_player_conditioning->type, "all") == 0);
                co_use_dc = (strcmp(conf.co_player_conditioning->type, "discount") == 0 ||
                             strcmp(conf.co_player_conditioning->type, "all") == 0);
            }

            co_player_net =
                init_drivenet(co_weights, num_co_players, env.dynamics_model, co_use_rc, co_use_ec, co_use_dc);
            printf("Co-player policy loaded with conditioning: rc=%d, ec=%d, dc=%d\n", co_use_rc, co_use_ec, co_use_dc);
        } else {
            printf("Warning: Could not load co-player policy from %s. Using main policy for all agents.\n",
                   actual_co_player_policy);
            num_ego_agents = env.active_agent_count;
            num_co_players = 0;
        }
    }

    // Extract co-player conditioning bounds from config
    float co_collision_lb = 0, co_collision_ub = 0;
    float co_offroad_lb = 0, co_offroad_ub = 0;
    float co_goal_lb = 0, co_goal_ub = 0;
    float co_entropy_lb = 0, co_entropy_ub = 0;
    float co_discount_lb = 0, co_discount_ub = 0;

    // Get conditioning dims directly from co_player_net to ensure consistency
    int coplayer_num_conditioning = (co_player_net != NULL) ? co_player_net->conditioning_dims : 0;

    if (conf.co_player_conditioning != NULL) {
        co_collision_lb = conf.co_player_conditioning->reward_collision_weight_lb;
        co_collision_ub = conf.co_player_conditioning->reward_collision_weight_ub;
        co_offroad_lb = conf.co_player_conditioning->reward_offroad_weight_lb;
        co_offroad_ub = conf.co_player_conditioning->reward_offroad_weight_ub;
        co_goal_lb = conf.co_player_conditioning->reward_goal_weight_lb;
        co_goal_ub = conf.co_player_conditioning->reward_goal_weight_ub;
        co_entropy_lb = conf.co_player_conditioning->entropy_weight_lb;
        co_entropy_ub = conf.co_player_conditioning->entropy_weight_ub;
        co_discount_lb = conf.co_player_conditioning->discount_weight_lb;
        co_discount_ub = conf.co_player_conditioning->discount_weight_ub;
    }

    // Load main (ego) policy
    Weights *weights = load_weights(policy_name);
    printf("Active agents in map: %d\n", env.active_agent_count);
    DriveNet *net = init_drivenet(weights, num_ego_agents, env.dynamics_model, use_rc, use_ec, use_dc);

    // Calculate frame count: k_scenarios * scenario_length for adaptive agents
    int scenario_length = env.scenario_length > 0 ? env.scenario_length : TRAJECTORY_LENGTH_DEFAULT;
    int k_scenarios = (k_scenarios_cli > 0) ? k_scenarios_cli : (conf.k_scenarios > 0 ? conf.k_scenarios : 1);
    int frame_count = k_scenarios * scenario_length;
    printf("Rendering %d scenarios x %d steps = %d total frames\n", k_scenarios, scenario_length, frame_count);
    char filename_topdown[256];
    char filename_agent[256];

    if (output_topdown != NULL && output_agent != NULL) {
        strcpy(filename_topdown, output_topdown);
        strcpy(filename_agent, output_agent);
    } else {
        char policy_base[256];
        strcpy(policy_base, policy_name);
        *strrchr(policy_base, '.') = '\0';

        char map[256];
        strcpy(map, basename((char *)map_name));
        *strrchr(map, '.') = '\0';

        char video_dir[256];
        sprintf(video_dir, "%s/video", policy_base);
        char mkdir_cmd[512];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", video_dir);
        system(mkdir_cmd);

        sprintf(filename_topdown, "%s/video/%s_topdown.mp4", policy_base, map);
        sprintf(filename_agent, "%s/video/%s_agent.mp4", policy_base, map);
    }

    bool render_topdown = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "topdown") == 0);
    bool render_agent = (strcmp(view_mode, "both") == 0 || strcmp(view_mode, "agent") == 0);

    printf("Rendering: %s\n", view_mode);

    int rendered_frames = 0;
    double startTime = GetTime();

    VideoRecorder topdown_recorder, agent_recorder;

    if (render_topdown) {
        if (!OpenVideo(&topdown_recorder, filename_topdown, img_width, img_height)) {
            CloseWindow();
            return -1;
        }
    }

    if (render_agent) {
        if (!OpenVideo(&agent_recorder, filename_agent, img_width, img_height)) {
            if (render_topdown)
                CloseVideo(&topdown_recorder);
            CloseWindow();
            return -1;
        }
    }

    // Calculate observation sizes per agent
    // ego_base_dim: 7 for CLASSIC dynamics, 10 for JERK dynamics
    int ego_base_dim = (env.dynamics_model == 1) ? 10 : 7; // 1 = JERK

    // Ego observation size (environment generates observations without conditioning for ego)
    int ego_obs_size =
        net->ego_dim + (MAX_AGENTS - 1) * PARTNER_FEATURES + MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES;

    // Co-player observation size (includes conditioning features)
    int coplayer_obs_size = ego_obs_size;
    if (co_player_net != NULL) {
        coplayer_obs_size = co_player_net->ego_dim + (MAX_AGENTS - 1) * PARTNER_FEATURES +
                            MAX_ROAD_SEGMENT_OBSERVATIONS * ROAD_FEATURES;
    }

    printf("Observation sizes: ego=%d, coplayer=%d, ego_base_dim=%d, coplayer_conditioning=%d\n", ego_obs_size,
           coplayer_obs_size, ego_base_dim, coplayer_num_conditioning);

    if (render_topdown) {
        printf("Recording topdown view...\n");
        for (int i = 0; i < frame_count; i++) {
            // Calculate current scenario (1-indexed for display)
            int current_scenario = (i / scenario_length) + 1;
            if (i % frame_skip == 0) {
                renderTopDownView(&env, client, map_width, map_height, 0, 0, 0, frame_count, NULL, show_human_logs,
                                  show_grid, img_width, img_height, zoom_in, current_scenario, k_scenarios);
                WriteFrame(&topdown_recorder, img_width, img_height);
                rendered_frames++;
            }
            forward_population(net, co_player_net, env.observations, (int *)env.actions, num_ego_agents, num_co_players,
                               ego_obs_size, coplayer_obs_size, ego_base_dim, co_use_rc, co_use_ec, co_use_dc,
                               co_collision_lb, co_collision_ub, co_offroad_lb, co_offroad_ub, co_goal_lb, co_goal_ub,
                               co_entropy_lb, co_entropy_ub, co_discount_lb, co_discount_ub);
            c_step(&env);
        }
    }

    if (render_agent) {
        c_reset(&env);
        printf("Recording agent view...\n");
        for (int i = 0; i < frame_count; i++) {
            int human_idx = env.active_agent_indices[env.human_agent_idx];
            if (env.entities[human_idx].respawn_count > 0) {
                break;
            }
            if (i % frame_skip == 0) {
                renderAgentView(&env, client, map_height, obs_only, lasers, show_grid);
                WriteFrame(&agent_recorder, img_width, img_height);
                rendered_frames++;
            }
            forward_population(net, co_player_net, env.observations, (int *)env.actions, num_ego_agents, num_co_players,
                               ego_obs_size, coplayer_obs_size, ego_base_dim, co_use_rc, co_use_ec, co_use_dc,
                               co_collision_lb, co_collision_ub, co_offroad_lb, co_offroad_ub, co_goal_lb, co_goal_ub,
                               co_entropy_lb, co_entropy_ub, co_discount_lb, co_discount_ub);
            c_step(&env);
        }
    }

    double endTime = GetTime();
    double elapsedTime = endTime - startTime;
    double writeFPS = (elapsedTime > 0) ? rendered_frames / elapsedTime : 0;

    printf("Wrote %d frames in %.2f seconds (%.2f FPS) to %s\n", rendered_frames, elapsedTime, writeFPS,
           filename_topdown);

    if (render_topdown) {
        CloseVideo(&topdown_recorder);
    }
    if (render_agent) {
        CloseVideo(&agent_recorder);
    }
    CloseWindow();

    free(client);
    free_allocated(&env);
    free_drivenet(net);
    free(weights);
    if (co_player_net != NULL) {
        free_drivenet(co_player_net);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    // Visualization-only parameters (not in [env] section)
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int show_human_logs = 0;
    int frame_skip = 1;
    int zoom_in = 0;
    const char *view_mode = "both";

    // File paths and num_maps (not in [env] section)
    const char *map_name = NULL;
    const char *policy_name = "resources/drive/puffer_drive_weights.bin";
    const char *co_player_policy_name = NULL;
    const char *output_topdown = NULL;
    const char *output_agent = NULL;
    const char *ini_file = "pufferlib/config/ocean/drive.ini";
    int num_maps = 1;
    int scenario_length_cli = -1;
    int k_scenarios_cli = -1;
    int max_controlled_agents_cli = -1;
    int use_rc = 0;
    int use_ec = 0;
    int use_dc = 0;
    int init_mode = 0;
    int control_mode = 0;
    int goal_behavior = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--show-grid") == 0) {
            show_grid = 1;
        } else if (strcmp(argv[i], "--obs-only") == 0) {
            obs_only = 1;
        } else if (strcmp(argv[i], "--lasers") == 0) {
            lasers = 1;
        } else if (strcmp(argv[i], "--log-trajectories") == 0) {
            show_human_logs = 1;
        } else if (strcmp(argv[i], "--frame-skip") == 0) {
            if (i + 1 < argc) {
                frame_skip = atoi(argv[i + 1]);
                i++;
                if (frame_skip <= 0) {
                    frame_skip = 1;
                }
            }
        } else if (strcmp(argv[i], "--zoom-in") == 0) {
            zoom_in = 1;
        } else if (strcmp(argv[i], "--view") == 0) {
            if (i + 1 < argc) {
                view_mode = argv[i + 1];
                i++;
                if (strcmp(view_mode, "both") != 0 && strcmp(view_mode, "topdown") != 0 &&
                    strcmp(view_mode, "agent") != 0) {
                    fprintf(stderr, "Error: --view must be 'both', 'topdown', or 'agent'\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: --view option requires a value (both/topdown/agent)\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--map-name") == 0) {
            if (i + 1 < argc) {
                map_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --map-name option requires a map file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--policy-name") == 0) {
            if (i + 1 < argc) {
                policy_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --policy-name option requires a policy file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--output-topdown") == 0) {
            if (i + 1 < argc) {
                output_topdown = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--output-agent") == 0) {
            if (i + 1 < argc) {
                output_agent = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--num-maps") == 0) {
            if (i + 1 < argc) {
                num_maps = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "--ini-file") == 0) {
            if (i + 1 < argc) {
                ini_file = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --ini-file option requires a file path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--k-scenarios") == 0) {
            if (i + 1 < argc) {
                k_scenarios_cli = atoi(argv[i + 1]);
                i++;
            } else {
                fprintf(stderr, "Error: --k-scenarios option requires a number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--max-controlled-agents") == 0) {
            if (i + 1 < argc) {
                max_controlled_agents_cli = atoi(argv[i + 1]);
                i++;
            } else {
                fprintf(stderr, "Error: --max-controlled-agents option requires a number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--co-player-policy") == 0) {
            if (i + 1 < argc) {
                co_player_policy_name = argv[i + 1];
                i++;
            } else {
                fprintf(stderr, "Error: --co-player-policy option requires a file path\n");
                return 1;
            }
        }
    }

    eval_gif(map_name, policy_name, show_grid, obs_only, lasers, show_human_logs, frame_skip, view_mode, output_topdown,
             output_agent, num_maps, zoom_in, ini_file, k_scenarios_cli, max_controlled_agents_cli,
             co_player_policy_name);
    return 0;
}
