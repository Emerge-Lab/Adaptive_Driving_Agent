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

void renderTopDownView(Drive *env, Client *client, int map_height, int obs, int lasers, int trajectories,
                       int frame_count, float *path, int show_human_logs, int show_grid, int img_width, int img_height,
                       int zoom_in) {
    BeginDrawing();

    // Top-down orthographic camera
    Camera3D camera = {0};

    if (zoom_in) {                                       // Zoom in on part of the map
        camera.position = (Vector3){0.0f, 0.0f, 500.0f}; // above the scene
        camera.target = (Vector3){0.0f, 0.0f, 0.0f};     // look at origin
        camera.fovy = map_height;
    } else { // Show full map
        camera.position = (Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 500.0f};
        camera.target = (Vector3){env->grid_map->top_left_x, env->grid_map->bottom_right_y, 0.0f};
        camera.fovy = 2 * map_height;
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


void parse_conditioning(const void* config_ptr, 
                        bool* reward, bool* entropy, bool* discount) {
    *reward = *entropy = *discount = false;
    const conditioning_config* cfg = (const conditioning_config*)config_ptr;
    if (!cfg || !cfg->type || strcmp(cfg->type, "none") == 0) return;

    bool all = strcmp(cfg->type, "all") == 0;
    *reward   = all || strcmp(cfg->type, "reward")   == 0;
    *entropy  = all || strcmp(cfg->type, "entropy")  == 0;
    *discount = all || strcmp(cfg->type, "discount") == 0;
}

static inline float rand_float_r(float lb, float ub, unsigned int* seed) {
    return lb + ((float)rand_r(seed) / RAND_MAX) * (ub - lb);
}

void get_conditioning_weights(const void* config_ptr,
                              float* offroad, float* collision, float* goal,
                              float* entropy, float* discount,
                              bool reward_cond, bool entropy_cond, bool discount_cond,
                              unsigned int* seed) {
    const conditioning_config* cfg = (const conditioning_config*)config_ptr;
    if (!cfg) return;

    if (reward_cond) {
        if (offroad)    *offroad    = rand_float_r(cfg->reward_offroad_weight_lb,    cfg->reward_offroad_weight_ub,    seed);
        if (collision)  *collision  = rand_float_r(cfg->reward_collision_weight_lb,  cfg->reward_collision_weight_ub,  seed);
        if (goal)       *goal       = rand_float_r(cfg->reward_goal_weight_lb,       cfg->reward_goal_weight_ub,       seed);
    }
    if (entropy_cond  && entropy)   *entropy  = rand_float_r(cfg->entropy_weight_lb,  cfg->entropy_weight_ub,  seed);
    if (discount_cond && discount)  *discount = rand_float_r(cfg->discount_weight_lb, cfg->discount_weight_ub, seed);
}

void copy_conditioned_observation(float* dest, const float* src, int max_obs,
                                  bool reward_cond, bool entropy_cond, bool discount_cond,
                                  float collision, float offroad, float goal,
                                  float entropy, float discount) {
    int cond_size = (reward_cond ? 3 : 0) + (entropy_cond ? 1 : 0) + (discount_cond ? 1 : 0);
    if (cond_size == 0) {
        memcpy(dest, src, max_obs * sizeof(float));
        return;
    }

    memcpy(dest, src, 7 * sizeof(float));
    int idx = 7;
    if (reward_cond)  { dest[idx++] = collision; dest[idx++] = offroad; dest[idx++] = goal; }
    if (entropy_cond)  dest[idx++] = entropy;
    if (discount_cond) dest[idx++] = discount;
    memcpy(&dest[idx], &src[7], (max_obs - 7) * sizeof(float));
}

void assign_population_actions(Drive* env, int ego_agent_id,
                               int* ego_actions, int* co_player_actions) {
    int* actions = (int*)env->actions;
    actions[ego_agent_id] = ego_actions[0];
    int co_idx = 0;
    for (int j = 0; j < env->active_agent_count; j++) {
        if (j != ego_agent_id) actions[j] = co_player_actions[co_idx++];
    }
}

// ---- Conditioning state bundled into a struct for cleaner passing ----

typedef struct {
    bool reward, entropy, discount;
    float offroad, collision, goal, entropy_w, discount_w;
    int size;
} CondState;

static CondState build_cond_state(const void* config_ptr, unsigned int* seed) {
    CondState s = {0};
    if (!config_ptr) return s;
    parse_conditioning(config_ptr, &s.reward, &s.entropy, &s.discount);
    s.size = (s.reward ? 3 : 0) + (s.entropy ? 1 : 0) + (s.discount ? 1 : 0);
    get_conditioning_weights(config_ptr,
                             &s.offroad, &s.collision, &s.goal, &s.entropy_w, &s.discount_w,
                             s.reward, s.entropy, s.discount, seed);
    return s;
}

static void prepare_population_obs(Drive* env, int ego_agent_id,
                                   float* ego_obs, float* co_obs,
                                   int max_obs, const CondState* ego, const CondState* co) {
    int ego_obs_size = max_obs + ego->size;
    int co_obs_size  = max_obs + co->size;

    copy_conditioned_observation(ego_obs,
                                 &env->observations[ego_agent_id * ego_obs_size], max_obs,
                                 ego->reward, ego->entropy, ego->discount,
                                 ego->collision, ego->offroad, ego->goal,
                                 ego->entropy_w, ego->discount_w);
    int co_idx = 0;
    for (int j = 0; j < env->active_agent_count; j++) {
        if (j == ego_agent_id) continue;
        copy_conditioned_observation(&co_obs[co_idx * co_obs_size],
                                     &env->observations[j * max_obs], max_obs,
                                     co->reward, co->entropy, co->discount,
                                     co->collision, co->offroad, co->goal,
                                     co->entropy_w, co->discount_w);
        co_idx++;
    }
}




// ---- Main eval function ----
// Car color indices matching the cars[] array: 0=Red,1=White,2=Blue,3=Yellow,4=Green,5=Grey
#define CAR_COLOR_EGO       2  // Blue
#define CAR_COLOR_CO_PLAYER 3  // Yellow

static void assign_agent_roles(Drive* env) {
    // Clear all agents first
    for (int i = 0; i < env->num_entities; i++)
        env->entities[i].is_co_player = false;
    // Mark ego agents
    for (int i = 0; i < env->num_ego_agents; i++)
        env->entities[env->ego_agent_ids[i]].is_co_player = false;
    // Mark co-players
    for (int i = 0; i < env->num_co_players; i++)
        env->entities[env->co_player_ids[i]].is_co_player = true;
}

static void assign_agent_colors(Drive* env) {
    for (int i = 0; i < env->num_ego_agents; i++)
        env->entities[env->ego_agent_ids[i]].color_idx = CAR_COLOR_EGO;
    for (int i = 0; i < env->num_co_players; i++)
        env->entities[env->co_player_ids[i]].color_idx = CAR_COLOR_CO_PLAYER;
}
static void run_render_loop(Drive* env, Client* client, VideoRecorder* recorder,
                            bool is_topdown, int frame_count, int frame_skip,
                            float map_height, int obs_only, int lasers,
                            int show_grid, int show_human_logs, int img_width, int img_height,
                            int zoom_in, DriveNet* ego_net, DriveNet* co_net, DriveNet* net,
                            float* ego_obs, float* co_obs,
                            int* ego_actions, int* co_actions, int max_obs,
                            int camera_agent) {
    for (int i = 0; i < frame_count; i++) {
        if (!is_topdown) {
            int cam_idx = env->active_agent_indices[camera_agent];
            if (env->entities[cam_idx].respawn_count > 0) break;
        }

        if (i % frame_skip == 0) {
            if (is_topdown)
                renderTopDownView(env, client, map_height, 0, 0, 0, frame_count,
                                  NULL, show_human_logs, show_grid, img_width, img_height, zoom_in);
            else
                renderAgentView(env, client, map_height, obs_only, lasers, show_grid);
            WriteFrame(recorder, img_width, img_height);
        }

        if (env->population_play) {
            for (int j = 0; j < env->num_ego_agents; j++)
                memcpy(&ego_obs[j * max_obs],
                       &env->observations[env->ego_agent_ids[j] * max_obs],
                       max_obs * sizeof(float));
            for (int j = 0; j < env->num_co_players; j++)
                memcpy(&co_obs[j * max_obs],
                       &env->observations[env->co_player_ids[j] * max_obs],
                       max_obs * sizeof(float));

            forward(ego_net, ego_obs, ego_actions);
            forward(co_net,  co_obs,  co_actions);

            for (int j = 0; j < env->num_ego_agents; j++)
                ((int*)env->actions)[env->ego_agent_ids[j]] = ego_actions[j];
            for (int j = 0; j < env->num_co_players; j++)
                ((int*)env->actions)[env->co_player_ids[j]] = co_actions[j];
        } else {
            forward(net, env->observations, (int*)env->actions);
        }
        c_step(env);
    }
}
static void conditioning_type_to_flags(const char* type, bool* use_rc, bool* use_ec, bool* use_dc) {
    *use_rc = *use_ec = *use_dc = false;
    if (!type || strcmp(type, "none") == 0) return;
    bool all = strcmp(type, "all") == 0;
    *use_rc = all || strcmp(type, "reward")   == 0;
    *use_ec = all || strcmp(type, "entropy")  == 0;
    *use_dc = all || strcmp(type, "discount") == 0;
}

int eval_gif(const char* map_name, const char* policy_name,
             int show_grid, int obs_only, int lasers, int show_human_logs,
             int frame_skip, const char* view_mode,
             const char* output_topdown, const char* output_agent,
             int num_maps, int zoom_in, int adaptive_driving_agent) {

    // Load config
    env_init_config conf = {0};
    const char* ini_file = "pufferlib/config/ocean/drive.ini";
    
    if (adaptive_driving_agent){
        ini_file = "pufferlib/config/ocean/adaptive.ini";
    }

    if (ini_parse(ini_file, handler, &conf) < 0) {
        fprintf(stderr, "Error: Could not load %s\n", ini_file);
        return -1;
    }

    // Resolve map path
    char map_buffer[100];
    if (!map_name) {
        srand(time(NULL));
        sprintf(map_buffer, "%s/map_%03d.bin", conf.map_dir, rand() % num_maps);
        map_name = map_buffer;
    }
    if (frame_skip <= 0) frame_skip = 1;

    // Validate files
    FILE* f;
    if (!(f = fopen(map_name,    "rb"))) { RAISE_FILE_ERROR(map_name);    } fclose(f);
    if (!(f = fopen(policy_name, "rb"))) { RAISE_FILE_ERROR(policy_name); } fclose(f);

    bool population_play = conf.co_player_enabled;

    bool use_rc = false, use_ec = false, use_dc = false;
    bool co_player_use_rc = false, co_player_use_ec = false, co_player_use_dc = false;

    if (conf.conditioning)
        conditioning_type_to_flags(conf.conditioning->type, &use_rc, &use_ec, &use_dc);
    if (conf.co_player_conditioning)
        conditioning_type_to_flags(conf.co_player_conditioning->type,
                                   &co_player_use_rc, &co_player_use_ec, &co_player_use_dc);

    const conditioning_config* c  = conf.conditioning;
    const conditioning_config* cp = conf.co_player_conditioning;

    map_name = "resources/drive/binaries/training/map_005.bin";

    Drive env = {
        .action_type              = conf.action_type,
        .dynamics_model           = conf.dynamics_model,
        .reward_vehicle_collision = conf.reward_vehicle_collision,
        .reward_offroad_collision = conf.reward_offroad_collision,
        .reward_goal              = conf.reward_goal,
        .reward_goal_post_respawn = conf.reward_goal_post_respawn,
        .goal_radius              = conf.goal_radius,
        .goal_behavior            = conf.goal_behavior,
        .goal_target_distance     = conf.goal_target_distance,
        .goal_speed               = conf.goal_speed,
        .dt                       = conf.dt,
        .scenario_length           = conf.scenario_length,
        .termination_mode         = conf.termination_mode,
        .collision_behavior       = conf.collision_behavior,
        .offroad_behavior         = conf.offroad_behavior,
        .init_steps               = conf.init_steps,
        .init_mode                = conf.init_mode,
        .control_mode             = conf.control_mode,
        .map_name                 = (char*)map_name,
        .max_controlled_agents    = conf.max_controlled_agents > 0 
                                    ? conf.max_controlled_agents : -1,
        .population_play          = population_play,
        // Ego conditioning
        .use_rc                   = use_rc,
        .use_ec                   = use_ec,
        .use_dc                   = use_dc,
        .collision_weight_lb      = c ? c->reward_collision_weight_lb : 0,
        .collision_weight_ub      = c ? c->reward_collision_weight_ub : 0,
        .offroad_weight_lb        = c ? c->reward_offroad_weight_lb   : 0,
        .offroad_weight_ub        = c ? c->reward_offroad_weight_ub   : 0,
        .goal_weight_lb           = c ? c->reward_goal_weight_lb      : 0,
        .goal_weight_ub           = c ? c->reward_goal_weight_ub      : 0,
        .entropy_weight_lb        = c ? c->entropy_weight_lb          : 0,
        .entropy_weight_ub        = c ? c->entropy_weight_ub          : 0,
        .discount_weight_lb       = c ? c->discount_weight_lb         : 0,
        .discount_weight_ub       = c ? c->discount_weight_ub         : 0,
        // Co-player conditioning
        .co_player_use_rc              = co_player_use_rc,
        .co_player_use_ec              = co_player_use_ec,
        .co_player_use_dc              = co_player_use_dc,
        .co_player_collision_weight_lb = cp ? cp->reward_collision_weight_lb : 0,
        .co_player_collision_weight_ub = cp ? cp->reward_collision_weight_ub : 0,
        .co_player_offroad_weight_lb   = cp ? cp->reward_offroad_weight_lb   : 0,
        .co_player_offroad_weight_ub   = cp ? cp->reward_offroad_weight_ub   : 0,
        .co_player_goal_weight_lb      = cp ? cp->reward_goal_weight_lb      : 0,
        .co_player_goal_weight_ub      = cp ? cp->reward_goal_weight_ub      : 0,
        .co_player_entropy_weight_lb   = cp ? cp->entropy_weight_lb          : 0,
        .co_player_entropy_weight_ub   = cp ? cp->entropy_weight_ub          : 0,
        .co_player_discount_weight_lb  = cp ? cp->discount_weight_lb         : 0,
        .co_player_discount_weight_ub  = cp ? cp->discount_weight_ub         : 0,
    };

    allocate(&env);

    if (env.active_agent_count == 0) {
        fprintf(stderr, "Error: Map %s has no controllable agents\n", map_name);
        free_allocated(&env);
        return -1;
    }

    // Assign ego and co-player IDs manually since my_init is not called here
    if (population_play) {
        // Pick a random ego agent from the active agents
        int ego_idx = rand() % env.active_agent_count;

        env.num_ego_agents = 1;
        env.ego_agent_ids = (int*)malloc(sizeof(int));
        env.ego_agent_ids[0] = env.active_agent_indices[ego_idx];

        env.num_co_players = env.active_agent_count - 1;
        env.co_player_ids = (int*)malloc(env.num_co_players * sizeof(int));
        int co_idx = 0;
        for (int i = 0; i < env.active_agent_count; i++) {
            if (i != ego_idx)
                env.co_player_ids[co_idx++] = env.active_agent_indices[i];
        }

        printf("Assigned ego_agent_id=%d, %d co-players\n",
               env.ego_agent_ids[0], env.num_co_players);
    }

    c_reset(&env);

    int camera_agent = population_play ? env.ego_agent_ids[0]
                                       : rand() % env.active_agent_count;

    if (population_play) {
        assign_agent_roles(&env);
        assign_agent_colors(&env);
        printf("Population play: %d ego agent(s), %d co-player(s)\n",
               env.num_ego_agents, env.num_co_players);
    }
    printf("Active agents: %d\n", env.active_agent_count);

    // Window setup
    float map_width  = env.grid_map->bottom_right_x - env.grid_map->top_left_x;
    float map_height = env.grid_map->top_left_y     - env.grid_map->bottom_right_y;
    int img_width  = (int)roundf(map_width  * 6.0f / 2.0f) * 2;
    int img_height = (int)roundf(map_height * 6.0f / 2.0f) * 2;
    printf("Map size: %.1fx%.1f\n", map_width, map_height);

    Client* client = (Client*)calloc(1, sizeof(Client));
    env.client = client;
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    SetTargetFPS(6000);
    InitWindow(img_width, img_height, "Puffer Drive");
    SetConfigFlags(FLAG_MSAA_4X_HINT);

    client->puffers    = LoadTexture("resources/puffers_128.png");
    client->cars[0]    = LoadModel("resources/drive/RedCar.glb");
    client->cars[1]    = LoadModel("resources/drive/WhiteCar.glb");
    client->cars[2]    = LoadModel("resources/drive/BlueCar.glb");
    client->cars[3]    = LoadModel("resources/drive/YellowCar.glb");
    client->cars[4]    = LoadModel("resources/drive/GreenCar.glb");
    client->cars[5]    = LoadModel("resources/drive/GreyCar.glb");
    client->cyclist    = LoadModel("resources/drive/cyclist.glb");
    client->pedestrian = LoadModel("resources/drive/pedestrian.glb");

    // Load networks and allocate buffers
    Weights*  weights     = load_weights(policy_name);
    Weights*  co_weights  = NULL;
    DriveNet* net         = NULL;
    DriveNet* ego_net     = NULL;
    DriveNet* co_net      = NULL;
    float*    ego_obs     = NULL;
    float*    co_obs      = NULL;
    int*      ego_actions = NULL;
    int*      co_actions  = NULL;
    int       max_obs     = (env.dynamics_model == JERK ? 10 : 7) + 7*63 + 7*200;

    if (population_play) {
        ego_net    = init_drivenet(weights,    env.num_ego_agents,  env.dynamics_model,
                                   env.use_rc,            env.use_ec,            env.use_dc);

        co_weights = load_weights("resources/drive/policies/co_player.bin");
        co_net     = init_drivenet(co_weights, env.num_co_players,  env.dynamics_model,
                                   env.co_player_use_rc,  env.co_player_use_ec,  env.co_player_use_dc);

        if (!weights || !ego_net || !co_weights || !co_net) {
            fprintf(stderr, "Error: Failed to load/init networks\n");
            CloseWindow(); return -1;
        }

        ego_obs     = calloc(env.num_ego_agents * max_obs, sizeof(float));
        co_obs      = calloc(env.num_co_players * max_obs, sizeof(float));
        ego_actions = calloc(env.num_ego_agents, sizeof(int));
        co_actions  = calloc(env.num_co_players, sizeof(int));

        if (!ego_obs || !co_obs || !ego_actions || !co_actions) {
            fprintf(stderr, "Error: Failed to allocate population play buffers\n");
            CloseWindow(); return -1;
        }
    } else {
        net = init_drivenet(weights, env.active_agent_count, env.dynamics_model,
                            env.use_rc, env.use_ec, env.use_dc);
        if (!net) {
            fprintf(stderr, "Error: Failed to init network\n");
            CloseWindow(); return -1;
        }
    }

    // Output paths
    char filename_topdown[256], filename_agent[256];
    if (output_topdown && output_agent) {
        strcpy(filename_topdown, output_topdown);
        strcpy(filename_agent,   output_agent);
    } else {
        char policy_base[256], map_base[256];
        strcpy(policy_base, policy_name);
        *strrchr(policy_base, '.') = '\0';
        strcpy(map_base, basename((char*)map_name));
        *strrchr(map_base, '.') = '\0';

        char mkdir_cmd[512];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s/video\"", policy_base);
        system(mkdir_cmd);

        sprintf(filename_topdown, "%s/video/%s_topdown.mp4", policy_base, map_base);
        sprintf(filename_agent,   "%s/video/%s_agent.mp4",   policy_base, map_base);
    }

    bool render_topdown = strcmp(view_mode, "both") == 0 || strcmp(view_mode, "topdown") == 0;
    bool render_agent   = strcmp(view_mode, "both") == 0 || strcmp(view_mode, "agent")   == 0;
    printf("Rendering: %s\n", view_mode);

    VideoRecorder topdown_rec, agent_rec;
    if (render_topdown && !OpenVideo(&topdown_rec, filename_topdown, img_width, img_height)) {
        CloseWindow(); return -1;
    }
    if (render_agent && !OpenVideo(&agent_rec, filename_agent, img_width, img_height)) {
        if (render_topdown) CloseVideo(&topdown_rec);
        CloseWindow(); return -1;
    }

    int    frame_count     = env.scenario_length > 0 ? env.scenario_length : TRAJECTORY_LENGTH_DEFAULT;
    int    rendered_frames = 0;
    double t0              = GetTime();

    if (render_topdown) {
        printf("Recording topdown view...\n");
        run_render_loop(&env, client, &topdown_rec, /*is_topdown=*/true,
                        frame_count, frame_skip, map_height, obs_only, lasers,
                        show_grid, show_human_logs, img_width, img_height, zoom_in,
                        ego_net, co_net, net, ego_obs, co_obs, ego_actions, co_actions,
                        max_obs, camera_agent);
        rendered_frames += frame_count / frame_skip;
    }

    if (render_agent) {
        c_reset(&env);  // colors already set, no need to reassign
        printf("Recording agent view...\n");
        run_render_loop(&env, client, &agent_rec, /*is_topdown=*/false,
                        frame_count, frame_skip, map_height, obs_only, lasers,
                        show_grid, show_human_logs, img_width, img_height, zoom_in,
                        ego_net, co_net, net, ego_obs, co_obs, ego_actions, co_actions,
                        max_obs, camera_agent);
        rendered_frames += frame_count / frame_skip;
    }

    double elapsed = GetTime() - t0;
    printf("Wrote %d frames in %.2fs (%.2f FPS) to %s\n",
           rendered_frames, elapsed, rendered_frames / elapsed, filename_topdown);

    if (render_topdown) CloseVideo(&topdown_rec);
    if (render_agent)   CloseVideo(&agent_rec);
    CloseWindow();

    if (population_play) {
        free_drivenet(ego_net);
        free_drivenet(co_net);
        free(co_weights);
        free(ego_obs);
        free(co_obs);
        free(ego_actions);
        free(co_actions);
        free(env.ego_agent_ids);
        free(env.co_player_ids);
    } else {
        free_drivenet(net);
    }
    free(weights);
    free(client);
    free_allocated(&env);
    return 0;
}

int main(int argc, char *argv[]) {
    // Visualization-only parameters (not in [env] section)
    int show_grid = 0;
    int obs_only = 0;
    int lasers = 0;
    int show_human_logs = 0;
    int frame_skip = 1;
    int zoom_in = 1;
    const char *view_mode = "both";
    int adaptive_driving_agent = 0;

    // File paths and num_maps (not in [env] section)
    const char *map_name = NULL;
    const char *policy_name = "resources/drive/puffer_drive_weights.bin";
    const char *output_topdown = NULL;
    const char *output_agent = NULL;
    int num_maps = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--show-grid") == 0) {
            show_grid = 1;
        } else if (strcmp(argv[i], "--obs-only") == 0) {
            obs_only = 1;
        } 
        else if (strcmp(argv[i], "--adaptive-driving-agent") == 0) {
            adaptive_driving_agent = 1;
        } 
        else if (strcmp(argv[i], "--lasers") == 0) {
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
        }
    }

    eval_gif(map_name, policy_name, show_grid, obs_only, lasers, show_human_logs, frame_skip, view_mode, output_topdown,
             output_agent, num_maps, zoom_in, adaptive_driving_agent);
    return 0;
}