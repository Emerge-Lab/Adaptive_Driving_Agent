## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)

import os
import sys
import glob
import ast
import time
import random
import shutil
import subprocess
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.utils
import pufferlib.utils

try:
    from pufferlib import _C
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation"
    )

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter

rich.traceback.install(show_locals=False)

import signal  # Aggressively exit on ctrl+c

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# ----------------------------------------------------------------------------
# Trial-mode debug logger. Set PUFFER_TRIAL_DEBUG_FILE=/path/to/log.jsonl to
# capture per-epoch GAE/cache/trial diagnostics as a stream of JSON records.
# No-op otherwise. Schema:
#   {"event": "...", "epoch": int, "step": int, ...event-specific fields}
# Events:
#   "rollout_end_of_epoch": rollout buffer summary at end of each eval phase
#   "gae_outer": pre-GAE stats + post-GAE advantage stats per training update
#   "gae_inner": per-minibatch stats inside the PPO update loop
#   "cache_reset": each cache-reset event during rollout (under episode-end)
# ----------------------------------------------------------------------------
import json as _json

_TRIAL_DEBUG_PATH = os.environ.get("PUFFER_TRIAL_DEBUG_FILE", "")
_TRIAL_DEBUG_ENABLED = bool(_TRIAL_DEBUG_PATH)
_TRIAL_DEBUG_FH = None


def _trial_debug_log(event, **data):
    """Append a JSON line to the trial-debug file. Cheap when disabled."""
    if not _TRIAL_DEBUG_ENABLED:
        return
    global _TRIAL_DEBUG_FH
    try:
        if _TRIAL_DEBUG_FH is None:
            _TRIAL_DEBUG_FH = open(_TRIAL_DEBUG_PATH, "a", buffering=1)
        rec = {"event": event, "ts": time.time(), **data}
        _TRIAL_DEBUG_FH.write(_json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


# Assume advantage kernel has been built if CUDA compiler is available
ADVANTAGE_CUDA = shutil.which("nvcc") is not None


class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config["seed"]
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        # Vecenv info
        self.adaptive_driving_agent = getattr(vecenv.driver_env, "env_name", None) == "adaptive_drive"
        if self.adaptive_driving_agent:
            if config.get("policy_architecture", "Recurrent") == "Recurrent":
                config["bptt_horizon"] = vecenv.driver_env.episode_length
            if config.get("policy_architecture", "Recurrent") == "Transformer":
                config["context_length"] = self.context_length = vecenv.driver_env.episode_length
                config["bptt_horizon"] = (
                    vecenv.driver_env.episode_length
                )  ## this is used downstream so you need to define it too
        else:
            if config.get("policy_architecture", "Recurrent") == "Transformer":
                self.context_length = config["context_length"]
                config["bptt_horizon"] = config["context_length"]

        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.population_play = getattr(vecenv, "population_play", False)
        if self.population_play:
            total_ego_agents = vecenv.num_ego_agents
            agents_for_calc = total_ego_agents
            if config.get("policy_architecture", "Recurrent") == "Recurrent":
                batch_size = vecenv.driver_env.num_ego_agents * config["bptt_horizon"] * vecenv.num_workers
            if config.get("policy_architecture", "Recurrent") == "Transformer":
                batch_size = vecenv.driver_env.num_ego_agents * config["context_length"] * vecenv.num_workers
            config["batch_size"] = batch_size  ## this is dynamic and based on ego agents
        else:
            agents_for_calc = total_agents

        # total_agents = vecenv.num_train_agents
        self.total_agents = total_agents

        # Experience
        if (
            config["batch_size"] == "auto"
            and config.get("bptt_horizon", "auto") == "auto"
            and config.get("context_length", "auto") == "auto"
        ):
            raise pufferlib.APIUsageError("Must specify batch_size, bptt_horizon, or context_length")
        elif config["batch_size"] == "auto":
            if config.get("policy_architecture", "Recurrent") == "Recurrent":
                config["batch_size"] = agents_for_calc * config["bptt_horizon"]
            elif config.get("policy_architecture", "Recurrent") == "Transformer":
                config["batch_size"] = agents_for_calc * config["context_length"]
        elif (
            config.get("bptt_horizon", "auto") == "auto"
            and config.get("policy_architecture", "Recurrent") == "Recurrent"
        ):
            config["bptt_horizon"] = config["batch_size"] // agents_for_calc
        elif (
            config.get("context_length", "auto") == "auto"
            and config.get("policy_architecture", "Recurrent") == "Transformer"
        ):
            config["context_length"] = config["batch_size"] // agents_for_calc

        batch_size = config["batch_size"]

        # Set horizon based on model type
        if config.get("policy_architecture", "Recurrent") == "Recurrent":
            horizon = config["bptt_horizon"]
        elif config.get("policy_architecture", "Recurrent") == "Transformer":
            horizon = config["context_length"]
        else:
            horizon = config.get("bptt_horizon", config.get("context_length", 1))

        config["bptt_horizon"] = horizon  # For backward compatibility

        segments = batch_size // horizon
        self.segments = segments
        self.horizon = horizon
        if not self.population_play:
            if total_agents > segments:
                raise pufferlib.APIUsageError(f"Total agents {total_agents} <= segments {segments}")

        device = config["device"]
        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == "cuda" and config["cpu_offload"],
            device="cpu" if config["cpu_offload"] else device,
        )
        self.actions = torch.zeros(
            segments,
            horizon,
            *atn_space.shape,
            device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype],
        )
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        # Per-step per-agent off-map flag (gb=3 B''). Same shape as terminals.
        # Used in training to (a) add a garbage-attention mask matching the
        # eval-time `garbage_mask`, and (b) gate PPO loss/entropy/value-loss
        # so limbo tuples don't contribute gradient.
        self.removed_history = torch.zeros(segments, horizon, device=device, dtype=torch.bool)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(
            total_ego_agents if self.population_play else total_agents, device=device, dtype=torch.int32
        )

        self.free_idx = total_agents
        self.render = config["render"]
        self.render_interval = config["render_interval"]

        # LSTM
        if config.get("rnn_name", "Recurrent") == "Recurrent":
            h = policy.hidden_size
            if self.population_play:
                n = vecenv.ego_agents_per_batch  # Use ego agents per batch
                num_chunks = total_ego_agents // n
                self.lstm_h = {i * n: torch.zeros(n, h, device=device) for i in range(num_chunks)}
                self.lstm_c = {i * n: torch.zeros(n, h, device=device) for i in range(num_chunks)}
            else:
                n = vecenv.agents_per_batch
                self.lstm_h = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}
                self.lstm_c = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}

        # TRANSFORMER
        if config.get("rnn_name", "Recurrent") == "Transformer":
            h = policy.hidden_size

            if self.population_play:
                n = vecenv.ego_agents_per_batch  # Use ego agents per batch
                num_chunks = total_ego_agents // n
                # Initialize transformer context buffers
                self.transformer_context = {i * n: torch.zeros(n, 0, h, device=device) for i in range(num_chunks)}
                self.transformer_position = {
                    i * n: torch.zeros(n, dtype=torch.long, device=device) for i in range(num_chunks)
                }
            else:
                n = vecenv.agents_per_batch
                num_chunks = total_agents // n
                # Initialize transformer context buffers
                self.transformer_context = {i * n: torch.zeros(n, 0, h, device=device) for i in range(num_chunks)}
                self.transformer_position = {
                    i * n: torch.zeros(n, dtype=torch.long, device=device) for i in range(num_chunks)
                }
            # K/V cache persistence for the streaming forward_eval path.
            # The model lazy-allocates k_cache and v_cache (list of per-layer
            # tensors) on first call when state.get("k_cache") is None. We
            # persist them here so the next rollout step finds the cache
            # already populated with past timesteps' projections — without
            # this, every step would lazy-allocate fresh empty caches and
            # the policy would attend only to the current step (silent bug
            # discovered 2026-05-02; broke all in-context-learning runs
            # prior to that). None initially → first call allocates.
            self.transformer_k_cache = {i * n: None for i in range(num_chunks)}
            self.transformer_v_cache = {i * n: None for i in range(num_chunks)}
            # B'' garbage_mask: per-agent per-cache-slot bool. The model marks
            # current slot True when env.removed[i]=1 (ego off-map). Attention
            # then excludes those slots. Lazy-allocated by the model on first
            # forward_eval — None here mirrors the k_cache pattern.
            self.transformer_garbage_mask = {i * n: None for i in range(num_chunks)}
            self.horizon = int(getattr(policy, "horizon", config.get("horizon", 0)) or 0)

        # Regression detector for the rnn_name plumbing bug — fires once.
        print(
            f"[VERIFY rnn_name] config.get('rnn_name')={config.get('rnn_name')!r}, "
            f"policy_architecture={config.get('policy_architecture')!r}, "
            f"has_lstm_h={hasattr(self, 'lstm_h')}, "
            f"has_transformer_k_cache={hasattr(self, 'transformer_k_cache')}",
            flush=True,
        )

        # Minibatching & gradient accumulation
        if self.adaptive_driving_agent:
            minibatch_size = config["minibatch_multiplier"] * horizon
            self.minibatch_size = minibatch_size
        else:
            minibatch_size = config["minibatch_size"]

        max_minibatch_size = config["max_minibatch_size"]
        self.minibatch_size = min(minibatch_size, max_minibatch_size)

        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f"minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly"
            )

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(f"batch_size {batch_size} must be >= minibatch_size {minibatch_size}")

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config["update_epochs"] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f"minibatch_size {self.minibatch_size} must be divisible by horizon {horizon}"
            )

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        if config["compile"]:
            self.policy = torch.compile(policy, mode=config["compile_mode"])
            if hasattr(policy, "forward_eval"):
                self.policy.forward_eval = torch.compile(policy.forward_eval, mode=config["compile_mode"])
            pufferlib.pytorch.sample_logits = torch.compile(
                pufferlib.pytorch.sample_logits, mode=config["compile_mode"]
            )

        # Optimizer
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        elif config["optimizer"] == "muon":
            from heavyball import ForeachMuon

            warnings.filterwarnings(action="ignore", category=UserWarning, module=r"heavyball.*")
            import heavyball.utils

            heavyball.utils.compile_mode = config["compile_mode"] if config["compile"] else None
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        self.optimizer = optimizer

        # ---- Resume optimizer / epoch / global_step from trainer_state.pt ----
        # When --load-model-path points at a checkpoint that has a sibling
        # trainer_state.pt (which the trainer writes alongside every model
        # checkpoint), restore optimizer momentum + counters so the resumed
        # run continues mid-cosine instead of warm-restarting at peak LR with
        # cold Adam moments.
        resume_epoch = 0
        resume_global_step = 0
        load_path = config.get("load_model_path")
        if load_path:
            state_path = os.path.join(os.path.dirname(load_path), "trainer_state.pt")
            if os.path.exists(state_path):
                try:
                    # weights_only=False: trainer_state.pt contains optimizer
                    # state (with class refs), not just tensors.
                    saved = torch.load(state_path, map_location=config["device"], weights_only=False)
                    optimizer.load_state_dict(saved["optimizer_state_dict"])
                    resume_epoch = int(saved.get("update", 0))
                    resume_global_step = int(saved.get("global_step", 0))
                    print(
                        f"[trainer-state] Resumed optimizer state from {state_path}\n"
                        f"[trainer-state]   epoch={resume_epoch}  global_step={resume_global_step}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[trainer-state] WARNING: could not load {state_path}: {e}", flush=True)

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        if self.population_play:
            # Under external_co_player_actions, driver_env.co_player_policy is
            # None (worker doesn't load it); the GPU-bound copy lives on the
            # vecenv as co_player_policy_func.
            export_co_player = getattr(vecenv, "co_player_policy_func", None) or vecenv.driver_env.co_player_policy
            co_player_path = f"resources/drive/{config['env']}_co_player.bin"
            export_args = {"env_name": config["env"], "path": co_player_path, **config}
            export(
                args=export_args,
                env_name=config["env"],
                vecenv=vecenv,
                policy=export_co_player,
                path=co_player_path,
                silent=True,
            )

        # ---- Centralized GPU co-player inference (when enabled) ------------
        self.external_co_player = bool(
            self.population_play
            and getattr(vecenv, "co_player_policy_func", None) is not None
            and config.get("env_config", {}).get("external_co_player_actions", False)
        )
        if self.external_co_player:
            co_policy = vecenv.co_player_policy_func.to(config["device"])
            co_policy.eval()
            self.co_player_policy = co_policy
            self.co_player_conditioning_dims = getattr(vecenv, "co_player_conditioning_dims", 0)
            # One state dict per worker (each worker holds its own slice of
            # co_players; the per-worker batch size is num_co_players_per_env).
            num_co_per_worker = vecenv.driver_env.num_co_players
            num_workers = vecenv.num_workers
            self._co_player_num_per_worker = num_co_per_worker
            # Per-worker state dicts. Start each as an empty dict so that
            # `forward_eval` lazily allocates the K/V cache on first call with
            # the correct (obs-derived) dtype — avoiding a cache dtype that
            # mismatches the layer-output dtype during reset_eval_state's
            # cache-prime path.
            self.co_player_state = {w: {} for w in range(num_workers)}
            print(
                f"[external co-player] Loaded co-player on {device}; "
                f"per-worker batch={num_co_per_worker}, conditioning_dims={self.co_player_conditioning_dims}, "
                f"num_workers={num_workers}",
                flush=True,
            )

        # Learning rate scheduler — if resuming, advance to the saved epoch
        # position so cosine annealing continues smoothly.
        epochs = config["total_timesteps"] // config["batch_size"]
        last_epoch_arg = -1
        if resume_epoch > 0:
            # CosineAnnealingLR requires `initial_lr` in each param_group when
            # last_epoch != -1; old optimizer states sometimes lack it.
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", config["learning_rate"])
            last_epoch_arg = resume_epoch - 1  # next .step() lands on resume_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, last_epoch=last_epoch_arg)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config["precision"]
        self.amp_context = contextlib.nullcontext()
        if config.get("amp", True) and config["device"] == "cuda":
            self.amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))
        if precision not in ("float32", "bfloat16"):
            raise pufferlib.APIUsageError(f"Invalid precision: {precision}: use float32 or bfloat16")

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = resume_epoch
        self.global_step = resume_global_step
        self.last_log_step = resume_global_step
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def _fill_external_co_player_actions(self, full_obs, info, env_id, dones, truncs):
        """Centralized co-player inference on GPU.

        Workers receive co_player actions via the shared `actions` SHM buffer
        (filled here) instead of running per-worker CPU forward passes.

        Args:
            full_obs: numpy obs from vecenv.recv(), shape
                (num_agents_per_recv_batch, *obs_shape).
            info: the raw info list from vecenv.recv(). Must contain a dict
                with key "_external_co_player_ids" giving the actual list of
                co-player agent indices (worker-local). Computing this from
                complement-of-ego_ids is wrong: many slots are padding or
                otherwise inactive, and forwarding garbage obs through them
                pollutes the shared KV cache.
            env_id: numpy array of agent indices for the current recv batch.
            dones, truncs: numpy bool per-agent done/trunc flags.
        """
        import numpy as np
        import torch

        device = self.config["device"]
        agents_per_worker = self.vecenv.agents_per_worker
        ego_agents_per_worker = getattr(self.vecenv, "ego_agents_per_worker", agents_per_worker)

        # Parse per-worker co_ids + reset flags from info (preserves order).
        # When vec.batch_size > 1, recv() returns N workers' obs+info stacked,
        # so info contains N dicts each with their own _external_co_player_ids.
        co_ids_per_worker = []
        reset_flags = []
        for item in info:
            if isinstance(item, dict) and "_external_co_player_ids" in item:
                co_ids_per_worker.append(list(item["_external_co_player_ids"]))
                reset_flags.append(bool(item.get("_external_reset_co_cache", False)))
        n_in_batch = len(co_ids_per_worker)
        if n_in_batch == 0:
            return

        base_worker_id = int(env_id[0]) // ego_agents_per_worker
        worker_ids = [base_worker_id + i for i in range(n_in_batch)]

        for i, w_id in enumerate(worker_ids):
            if reset_flags[i]:
                self.co_player_state[w_id] = {}

        # Build batched co_obs across workers. cum[] holds per-worker slice
        # offsets in the batched tensor for distributing actions afterwards.
        co_obs_width = getattr(self.vecenv.driver_env, "_c_obs_dim", None)
        parts = []
        cum = [0]
        for i in range(n_in_batch):
            co_ids = co_ids_per_worker[i]
            worker_obs = full_obs[i * agents_per_worker : (i + 1) * agents_per_worker]
            wco = worker_obs[co_ids] if co_obs_width is None else worker_obs[co_ids, :co_obs_width]
            parts.append(wco)
            cum.append(cum[-1] + len(co_ids))
        if cum[-1] == 0:
            return

        co_obs_np = np.concatenate(parts, axis=0)
        co_obs = torch.as_tensor(co_obs_np, device=device)
        if self.co_player_conditioning_dims > 0:
            cond_shm = self.vecenv.co_player_conditioning
            cond_parts = [cond_shm[worker_ids[i], : len(co_ids_per_worker[i]), :] for i in range(n_in_batch)]
            cond_np = np.concatenate(cond_parts, axis=0)
            cond = torch.as_tensor(cond_np, device=device, dtype=co_obs.dtype)
            from pufferlib.ocean.drive import binding as _b

            base_ego_dim = (
                _b.EGO_FEATURES_JERK if self.vecenv.driver_env.dynamics_model == "jerk" else _b.EGO_FEATURES_CLASSIC
            )
            co_obs = torch.cat([co_obs[:, :base_ego_dim], cond, co_obs[:, base_ego_dim:]], dim=1)

        # Merge per-worker KV caches into one batched cache so the policy
        # runs a single forward over all workers' co-players. Caches stay
        # per-worker in storage — they're only briefly stacked for the call.
        states = [self.co_player_state[w_id] for w_id in worker_ids]
        batched_state = {}
        if all("k_cache" in s and s["k_cache"] is not None for s in states):
            n_layers = len(states[0]["k_cache"])
            batched_state["k_cache"] = [
                torch.cat([s["k_cache"][li] for s in states], dim=0) for li in range(n_layers)
            ]
            batched_state["v_cache"] = [
                torch.cat([s["v_cache"][li] for s in states], dim=0) for li in range(n_layers)
            ]
        if all("garbage_mask" in s and s["garbage_mask"] is not None for s in states):
            batched_state["garbage_mask"] = torch.cat([s["garbage_mask"] for s in states], dim=0)
        if "transformer_position" in states[0]:
            batched_state["transformer_position"] = states[0]["transformer_position"]

        with torch.no_grad():
            logits, _ = self.co_player_policy.forward_eval(co_obs, batched_state)

        # Split updated state back to per-worker stores along the batch dim.
        for i, w_id in enumerate(worker_ids):
            start, end = cum[i], cum[i + 1]
            ns = {}
            if "k_cache" in batched_state:
                ns["k_cache"] = [k[start:end] for k in batched_state["k_cache"]]
                ns["v_cache"] = [v[start:end] for v in batched_state["v_cache"]]
            if "garbage_mask" in batched_state:
                ns["garbage_mask"] = batched_state["garbage_mask"][start:end]
            if "transformer_position" in batched_state:
                ns["transformer_position"] = batched_state["transformer_position"]
            self.co_player_state[w_id] = ns

        if isinstance(logits, tuple):
            co_action = torch.cat([l.argmax(dim=-1, keepdim=True) for l in logits], dim=-1)
        else:
            co_action = logits.argmax(dim=-1)
        co_action_np = co_action.cpu().numpy().reshape(cum[-1], -1)

        for i, w_id in enumerate(worker_ids):
            co_ids = co_ids_per_worker[i]
            start, end = cum[i], cum[i + 1]
            co_action_view = self.vecenv.actions[w_id]
            co_action_view[co_ids] = co_action_np[start:end].reshape((len(co_ids),) + co_action_view.shape[1:])

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        config = self.config
        device = config["device"]

        # Reset hidden states for both RNN and Transformer
        if config.get("rnn_name", "Recurrent") == "Recurrent":
            for k in self.lstm_h:
                self.lstm_h[k] = torch.zeros(self.lstm_h[k].shape, device=device)
                self.lstm_c[k] = torch.zeros(self.lstm_c[k].shape, device=device)

        if config.get("rnn_name", "Recurrent") == "Transformer":
            h = self.policy.hidden_size
            for k in self.transformer_context:
                n = self.transformer_context[k].shape[0]
                # Pre-allocate full buffer instead of empty
                self.transformer_context[k] = torch.zeros(n, self.horizon, h, device=device)
                self.transformer_position[k] = torch.zeros(1, dtype=torch.long, device=device)
                # Drop K/V cache so the model lazy-allocates fresh on
                # the first forward_eval call of this rollout.
                self.transformer_k_cache[k] = None
                self.transformer_v_cache[k] = None
                # B'' garbage_mask MUST also reset to None — otherwise stale
                # limbo marks from the prior rollout exclude valid slots in
                # the freshly-allocated cache, breaking rollout/replay parity
                # under gb=3. Under gb=0 (removed always 0) this is a no-op.
                self.transformer_garbage_mask[k] = None

        self.full_rows = 0
        # Hold autocast active across the whole rollout so torch.compile
        # doesn't see autocast GLOBAL_STATE flip between forward_eval calls
        # (would recompile every step under reduce-overhead).
        self.amp_context.__enter__()
        while self.full_rows < self.segments:
            profile("env", epoch)
            # print(".", end="", flush=True)  # Workaround: visible I/O prevents multiprocessing deadlock
            o, r, d, t, info, env_id, mask = self.vecenv.recv()
            # print(f"o shape is {o.shape}", flush = True)
            if self.population_play:
                batch_size = self.vecenv.batch_size
                # Filter info to get only the ego_ids lists (not the metrics dicts)
                ego_ids_per_env = [item for item in info if isinstance(item, list)]

                if self.external_co_player:
                    # Run co-player forward on GPU before the ego-only slicing
                    # below (we need the FULL obs array to extract co-player obs).
                    self._fill_external_co_player_actions(o, info, env_id, d, t)

                if batch_size > 1:
                    total_agents = len(o)
                    num_agents_per_env = total_agents // batch_size

                    # Create flat ego_ids by adding batch offset
                    flat_ego_ids = []
                    for env_idx in range(batch_size):
                        ego_ids = ego_ids_per_env[env_idx]
                        offset = env_idx * num_agents_per_env
                        flat_ego_ids.extend([int(idx) + offset for idx in ego_ids])

                    # Simply index with the flat ego_ids
                    o = o[flat_ego_ids]
                    r = r[flat_ego_ids]
                    d = d[flat_ego_ids]
                    t = t[flat_ego_ids]
                else:
                    ego_ids = ego_ids_per_env[0]  # Single environment
                    ego_ids = [int(idx) for idx in ego_ids]  # Convert to int
                    o = o[ego_ids]
                    r = r[ego_ids]
                    d = d[ego_ids]
                    t = t[ego_ids]

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            # KV cache + PE reset gate on `d` (terminals) only. Trial
            # boundaries (`t`, truncations) keep the cache so the policy
            # adapts across trials within an episode. See
            # docs/src/trial_mode.md.
            done_mask = d
            self.global_step += int(mask.sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device, non_blocking=True)
            r = torch.as_tensor(r, device=device)
            d = torch.as_tensor(d, device=device)

            profile("eval_forward", epoch)
            with torch.no_grad():
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )
                # Get appropriate batch key for state lookup
                if self.population_play:
                    batch_size = self.vecenv.ego_agents_per_batch
                else:
                    batch_size = self.vecenv.agents_per_batch
                state_key = (env_id.start // batch_size) * batch_size

                if config.get("rnn_name", "Recurrent") == "Recurrent":
                    state["lstm_h"] = self.lstm_h[state_key]
                    state["lstm_c"] = self.lstm_c[state_key]

                if config.get("rnn_name", "Recurrent") == "Transformer":
                    state["transformer_context"] = self.transformer_context[state_key]
                    state["transformer_position"] = self.transformer_position[state_key]
                    # K/V cache for streaming attention. None on the first
                    # call → model lazy-allocates (and resets pos to 0).
                    # Subsequent calls reuse the populated cache, which is
                    # the whole point: each step appends one new K/V slot
                    # and the policy attends over the full accumulated past.
                    state["k_cache"] = self.transformer_k_cache[state_key]
                    state["v_cache"] = self.transformer_v_cache[state_key]
                    state["garbage_mask"] = self.transformer_garbage_mask[state_key]
                    # B'' off-map flag. The model uses this to (a) mark the
                    # current cache slot as garbage in garbage_mask, and
                    # (b) exclude existing garbage slots from this step's
                    # attention. Unified flat (num_agents,) view exposed by
                    # the vec backend: Multiprocessing returns a SHM view
                    # so worker writes are visible; Serial/native return
                    # the in-process numpy array. None or all-False if the
                    # env doesn't expose `removed` (e.g. non-gb=3 modes).
                    rem_buf = getattr(self.vecenv, "removed", None)
                    if rem_buf is None:
                        rem_buf = getattr(self.vecenv.driver_env, "removed", None)
                    if rem_buf is not None:
                        rem_np = np.asarray(rem_buf)[env_id]
                        state["removed"] = torch.as_tensor(rem_np, device=device, dtype=torch.bool)
                    # Note: terminals not needed for eval since we're doing single-step inference

                # print(".", end="", flush=True)  # Prevents multiprocessing deadlock
                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)

            profile("eval_copy", epoch)
            with torch.no_grad():
                # Update hidden states after forward pass
                if config.get("rnn_name", "Recurrent") == "Recurrent":
                    if self.population_play:
                        batch_size = self.vecenv.ego_agents_per_batch
                    else:
                        batch_size = self.vecenv.agents_per_batch

                    lstm_key = (env_id.start // batch_size) * batch_size
                    self.lstm_h[lstm_key] = state["lstm_h"]
                    self.lstm_c[lstm_key] = state["lstm_c"]

                if config.get("rnn_name", "Recurrent") == "Transformer":
                    if self.population_play:
                        batch_size = self.vecenv.ego_agents_per_batch
                    else:
                        batch_size = self.vecenv.agents_per_batch

                    transformer_key = (env_id.start // batch_size) * batch_size
                    self.transformer_context[transformer_key] = state["transformer_context"]
                    self.transformer_position[transformer_key] = state["transformer_position"]
                    # Persist the K/V cache the model just wrote/updated so
                    # the next forward_eval call sees the accumulated past.
                    # state.get(...) is defensive: model may not have set
                    # these if it took the legacy path.
                    self.transformer_k_cache[transformer_key] = state.get("k_cache")
                    self.transformer_v_cache[transformer_key] = state.get("v_cache")
                    self.transformer_garbage_mask[transformer_key] = state.get("garbage_mask")

                    # Episode-boundary reset. pos is a shared (1,) scalar
                    # across the chunk; cache rows are per-agent. Filter
                    # done indices against the cache's batch dim, not the
                    # pos buffer's (1,) shape.
                    if done_mask.any():
                        done_indices = torch.where(torch.from_numpy(done_mask))[0]
                        if len(done_indices) > 0:
                            batch_start_in_group = env_id.start % batch_size
                            global_indices = batch_start_in_group + done_indices
                            kc = self.transformer_k_cache[transformer_key]
                            vc = self.transformer_v_cache[transformer_key]
                            cache_batch_dim = kc[0].shape[0] if kc is not None else 0
                            valid_mask = global_indices < cache_batch_dim
                            valid_indices = global_indices[valid_mask]
                            if len(valid_indices) > 0:
                                self.transformer_position[transformer_key][:] = 0
                                if kc is not None and vc is not None:
                                    for c in kc:
                                        c[valid_indices] = 0
                                    for c in vc:
                                        c[valid_indices] = 0
                                gm = self.transformer_garbage_mask[transformer_key]
                                if gm is not None:
                                    gm[valid_indices] = False
                                if _TRIAL_DEBUG_ENABLED:
                                    # At this point d/t may be torch CUDA tensors
                                    # (converted earlier in this block). Use done_mask
                                    # (still numpy) for the boundary count.
                                    _trial_debug_log(
                                        "cache_reset",
                                        epoch=int(self.epoch),
                                        step=int(self.global_step),
                                        env_id_start=int(env_id.start),
                                        env_id_stop=int(env_id.stop),
                                        n_done=int(len(valid_indices)),
                                        done_mask_sum=int(np.asarray(done_mask).sum()),
                                    )
                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1 + self.ep_indices[env_id.stop - 1].item())

                if config["cpu_offload"]:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                # Persist truncations for GAE bootstrap-stop. Stays out of
                # state["terminals"] so attention/PE span trial boundaries.
                t_tensor = torch.as_tensor(t, device=device).float()
                self.truncations[batch_rows, l] = t_tensor
                self.values[batch_rows, l] = value.flatten()
                # Persist per-step `removed` flag for train/eval mask parity.
                # During training we (a) add a garbage-attention mask matching
                # eval's `garbage_mask`, and (b) gate PPO losses so limbo
                # tuples don't contribute gradient.
                rem_buf = getattr(self.vecenv, "removed", None)
                if rem_buf is None:
                    rem_buf = getattr(self.vecenv.driver_env, "removed", None)
                if rem_buf is not None:
                    rem_step = torch.as_tensor(
                        np.asarray(rem_buf)[env_id], device=device, dtype=torch.bool
                    )
                    self.removed_history[batch_rows, l] = rem_step
                self.ep_lengths[env_id] += 1
                # Use appropriate horizon based on model type
                horizon = (
                    config.get("context_length")
                    if config.get("policy_architecture", "Recurrent") == "Transformer"
                    else config["bptt_horizon"]
                )
                if l + 1 >= horizon:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config["device"]).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile("env", epoch)
            self.vecenv.send(action)

        # Exit the autocast context that wraps the rollout loop.
        self.amp_context.__exit__(None, None, None)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents

        if self.population_play:
            total_agents = self.vecenv.num_ego_agents
        else:
            total_agents = self.total_agents

        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
        config = self.config
        device = config["device"]

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)

            if hasattr(self.vecenv.driver_env, "discount_conditioned") and self.vecenv.driver_env.discount_conditioned:
                if (
                    hasattr(self.vecenv.driver_env, "dynamics_model")
                    and self.vecenv.driver_env.dynamics_model == "jerk"
                ):
                    disc_idx = 12  # EGO_FEATURES_JERK (was 10 before lane features)
                else:
                    disc_idx = 9  # EGO_FEATURES_CLASSIC (was 7 before lane features)

                if self.vecenv.driver_env.reward_conditioned:
                    disc_idx += 3
                if self.vecenv.driver_env.entropy_conditioned:
                    disc_idx += 1
                gammas = self.observations[:, 0, disc_idx].to(device).contiguous()
            else:
                gammas = torch.full((self.segments,), config["gamma"], device=device, dtype=torch.float32)

            # GAE bootstrap-stop = terminals ∨ truncations ∨ removed.
            # - terminals: episode boundary (full reset)
            # - truncations: trial boundary under gb=3 (world resets, KV cache persists)
            # - removed: ego is off-map (limbo). V at limbo is computed from
            #   garbage (INVALID_POSITION) obs; bootstrapping from it would
            #   poison the prior step's advantage. Treat each limbo slot as
            #   a value-chain cut.
            #
            # Env var GAE_BOOTSTRAP_AT_TRUNCATIONS=1 restores standard GAE:
            # bootstrap V(s_{t+1}) across truncations, only cut at true
            # terminals + removed. Tests whether the trial-end bootstrap cut
            # is the cause of low gb=3 scores.
            if os.environ.get("GAE_BOOTSTRAP_AT_TRUNCATIONS", "0") == "1":
                bootstrap_stop = (
                    self.terminals + self.removed_history.float()
                ).clamp(max=1.0)
            else:
                bootstrap_stop = (
                    self.terminals + self.truncations + self.removed_history.float()
                ).clamp(max=1.0)
            if _TRIAL_DEBUG_ENABLED:
                _trial_debug_log(
                    "gae_outer_pre",
                    epoch=int(self.epoch),
                    minibatch=int(mb),
                    step=int(self.global_step),
                    terminals_sum=float(self.terminals.sum().item()),
                    truncations_sum=float(self.truncations.sum().item()),
                    bootstrap_stop_sum=float(bootstrap_stop.sum().item()),
                    bootstrap_overlap=float(
                        torch.minimum(self.terminals, self.truncations).sum().item()
                    ),
                    values_mean=float(self.values.mean().item()),
                    values_std=float(self.values.std().item()),
                    rewards_mean=float(self.rewards.mean().item()),
                    rewards_sum=float(self.rewards.sum().item()),
                )
            advantages = compute_puff_advantage(
                self.values,
                self.rewards,
                bootstrap_stop,
                self.ratio,
                advantages,
                gammas,
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            if _TRIAL_DEBUG_ENABLED:
                adv_flat = advantages.flatten()
                _trial_debug_log(
                    "gae_outer_post",
                    epoch=int(self.epoch),
                    minibatch=int(mb),
                    step=int(self.global_step),
                    adv_mean=float(adv_flat.mean().item()),
                    adv_std=float(adv_flat.std().item()),
                    adv_min=float(adv_flat.min().item()),
                    adv_max=float(adv_flat.max().item()),
                    adv_nan_count=int(torch.isnan(adv_flat).sum().item()),
                    adv_inf_count=int(torch.isinf(adv_flat).sum().item()),
                )

            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            # When cpu_offload=True, self.observations lives on CPU but `idx`
            # is on the training device (GPU). PyTorch refuses cross-device
            # fancy indexing, so move the index to CPU for the gather, then
            # ship the resulting minibatch to the device. Buffer was allocated
            # with pin_memory=True (see __init__) so the H2D copy is fast.
            if config["cpu_offload"]:
                mb_obs = self.observations[idx.cpu()].to(device, non_blocking=True)
            else:
                mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_removed = self.removed_history[idx]  # (B, T) bool — 1 = limbo step
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile("train_forward", epoch)

            # Handle observation reshaping based on model type
            if (
                not config.get("rnn_name", "Recurrent") == "Recurrent"
                and not config.get("rnn_name", "Recurrent") == "Transformer"
            ):
                # Flatten for non-recurrent models
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
            )

            # Add appropriate state based on model type
            if config.get("rnn_name", "Recurrent") == "Recurrent":
                state["lstm_h"] = None
                state["lstm_c"] = None
            elif config.get("rnn_name", "Recurrent") == "Transformer":
                state["transformer_context"] = None
                state["transformer_position"] = None
                state["terminals"] = mb_terminals  # For episode boundary masking
                state["removed"] = mb_removed      # Train/eval mask parity (gb=3)

            logits, newvalue = self.policy(mb_obs, state)

            # Handle action sampling based on observation shape
            if (
                config.get("rnn_name", "Recurrent") == "Recurrent"
                or config.get("rnn_name", "Recurrent") == "Transformer"
            ):
                # Add this right before calling sample_logits
                if isinstance(logits, tuple):
                    logits = logits[0]
                actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)
            else:
                # Need to flatten actions for non-recurrent models
                actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(
                    logits,
                    action=mb_actions.reshape(-1, *mb_actions.shape[2:]) if len(mb_actions.shape) > 2 else mb_actions,
                )

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            # Limbo importance ratios are computed from garbage obs / actions
            # and would poison the outer GAE's v-trace coefficients on the
            # next minibatch. Preserve the existing ratio at limbo positions.
            ratio_to_store = ratio.detach()
            if mb_removed is not None:
                ratio_to_store = torch.where(mb_removed, self.ratio[idx], ratio_to_store)
            self.ratio[idx] = ratio_to_store

            with torch.no_grad():
                # Mask limbo steps from diagnostics too so values aren't
                # inflated by garbage tuples (mb_removed will be available
                # in scope by the time these are reported; safe to reference).
                _diag_mask = (~mb_removed).to(logratio.dtype)
                _diag_n = _diag_mask.sum().clamp(min=1.0)
                old_approx_kl = ((-logratio) * _diag_mask).sum() / _diag_n
                approx_kl = (((ratio - 1) - logratio) * _diag_mask).sum() / _diag_n
                clipfrac = (((ratio - 1.0).abs() > config["clip_coef"]).float() * _diag_mask).sum() / _diag_n

            # Parity probe: at epoch 0 mb 0, replay batch-row-0 through the
            # EVAL-PATH forward step by step and compare logits per position
            # against the train-path logits. Gated by PUFFER_PARITY_PROBE=1.
            if (
                self.epoch == 0
                and mb == 0
                and os.environ.get("PUFFER_PARITY_PROBE", "0") == "1"
            ):
                import pickle
                B0, T0 = mb_obs.shape[0], mb_obs.shape[1]
                # Pick batch row 0
                b0_obs = mb_obs[0:1].to(device)  # (1, T, obs_dim)
                b0_removed = mb_removed[0:1].to(device)  # (1, T) bool
                b0_terminals = mb_terminals[0:1].to(device)  # (1, T) float
                eval_state = dict(
                    transformer_context=None,
                    transformer_position=None,
                    k_cache=None,
                    v_cache=None,
                    garbage_mask=None,
                )
                eval_logits_per_step = []
                # Unwrap compiled policy if necessary. Keep model in TRAIN
                # mode (no .eval() toggle) to match original rollout behavior.
                _pol = getattr(self.policy, "_orig_mod", self.policy)
                with torch.no_grad():
                    for t in range(T0):
                        # On terminal at position t, reset cache like pufferl does
                        # for the rollout. Under our gb=3 setup, terminal only fires
                        # at position 401 (end of episode), not within.
                        if t > 0 and float(b0_terminals[0, t - 1].item()) > 0.5:
                            eval_state["k_cache"] = None
                            eval_state["v_cache"] = None
                            eval_state["transformer_position"] = torch.zeros(1, dtype=torch.long, device=device)
                            eval_state["garbage_mask"] = None
                        obs_t = b0_obs[:, t, :]  # (1, obs_dim)
                        eval_state["removed"] = b0_removed[:, t]
                        l_t, _v = _pol.forward_eval(obs_t, eval_state)
                        if isinstance(l_t, tuple):
                            l_t = l_t[0]
                        eval_logits_per_step.append(l_t.detach().cpu())
                eval_logits = torch.stack(eval_logits_per_step, dim=1).squeeze(0)  # (T, A)

                # Train logits for batch row 0
                if isinstance(logits, tuple):
                    _lt = logits[0]
                else:
                    _lt = logits
                # _lt is shape (B*T, A) usually
                train_logits_b0 = _lt.detach().cpu().view(B0, T0, -1)[0]  # (T, A)

                _probe_path = "/scratch/mmk9418/projects/Adaptive_Driving_Agent/logs/parity_probe_v2.pkl"
                with open(_probe_path, "wb") as _f:
                    pickle.dump(
                        {
                            "logits_train_b0": train_logits_b0,
                            "logits_eval_b0": eval_logits,
                            "mb_actions_b0": mb_actions[0].cpu(),
                            "mb_logprobs_b0": mb_logprobs[0].cpu(),
                            "newlogprob_b0": newlogprob[0].cpu(),
                            "mb_removed_b0": b0_removed[0].cpu(),
                            "mb_terminals_b0": b0_terminals[0].cpu(),
                            "mb_truncations_b0": mb_truncations[0].cpu(),
                        },
                        _f,
                    )
                print(f"[parity_probe_v2] dumped batch-row-0 train+eval logits to {_probe_path}", flush=True)

            adv = advantages[idx]
            if hasattr(self.vecenv.driver_env, "discount_conditioned") and self.vecenv.driver_env.discount_conditioned:
                mb_gammas = gammas[idx]
            else:
                mb_gammas = torch.full((len(idx),), config["gamma"], device=device, dtype=torch.float32)

            # Recompute advantages with new ratios — bootstrap-stop is
            # terminals OR truncations OR removed (see outer GAE call comment).
            mb_bootstrap_stop = (
                mb_terminals + mb_truncations + mb_removed.float()
            ).clamp(max=1.0)
            if _TRIAL_DEBUG_ENABLED:
                # Split ratio by mb_removed to localize parity bug.
                active = ~mb_removed
                limbo = mb_removed
                ratio_active = ratio[active] if active.any() else ratio.new_empty(0)
                ratio_limbo = ratio[limbo] if limbo.any() else ratio.new_empty(0)
                _trial_debug_log(
                    "gae_inner",
                    epoch=int(self.epoch),
                    minibatch=int(mb),
                    step=int(self.global_step),
                    mb_terminals_sum=float(mb_terminals.sum().item()),
                    mb_truncations_sum=float(mb_truncations.sum().item()),
                    mb_bootstrap_sum=float(mb_bootstrap_stop.sum().item()),
                    ratio_mean=float(ratio.mean().item()),
                    ratio_min=float(ratio.min().item()),
                    ratio_max=float(ratio.max().item()),
                    active_n=int(active.sum().item()),
                    active_ratio_mean=float(ratio_active.mean().item()) if ratio_active.numel() else 0.0,
                    active_ratio_min=float(ratio_active.min().item()) if ratio_active.numel() else 0.0,
                    active_ratio_max=float(ratio_active.max().item()) if ratio_active.numel() else 0.0,
                    active_ratio_std=float(ratio_active.std().item()) if ratio_active.numel() > 1 else 0.0,
                    limbo_n=int(limbo.sum().item()),
                    limbo_ratio_mean=float(ratio_limbo.mean().item()) if ratio_limbo.numel() else 0.0,
                    limbo_ratio_min=float(ratio_limbo.min().item()) if ratio_limbo.numel() else 0.0,
                    limbo_ratio_max=float(ratio_limbo.max().item()) if ratio_limbo.numel() else 0.0,
                    limbo_ratio_std=float(ratio_limbo.std().item()) if ratio_limbo.numel() > 1 else 0.0,
                    approx_kl=float(approx_kl.item()),
                    clipfrac=float(clipfrac.item()),
                    adv_mean_pre=float(adv.mean().item()),
                    adv_std_pre=float(adv.std().item()),
                )
            adv = compute_puff_advantage(
                mb_values,
                mb_rewards,
                mb_bootstrap_stop,
                ratio,
                adv,
                mb_gammas,
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            # Per-step validity mask: 1 where the agent was ACTIVE (not limbo),
            # 0 where removed=1 (off-map). All per-sample losses are weighted
            # by this and normalized by the count of valid samples, so limbo
            # tuples contribute zero gradient. mb_removed has shape (B, T)
            # matching the per-step losses below.
            valid_mask = (~mb_removed).to(adv.dtype)            # (B, T)
            n_valid = valid_mask.sum().clamp(min=1.0)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = (torch.max(pg_loss1, pg_loss2) * valid_mask).sum() / n_valid

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * valid_mask).sum() / n_valid

            # Entropy-weighted loss if entropy conditioning is enabled.
            # NOTE: entropy comes back from sample_logits FLAT — shape (B*T,)
            # — while valid_mask is (B, T). Flatten valid_mask once for these
            # mults so we don't crash on broadcast.
            valid_mask_flat = valid_mask.reshape(-1)
            n_valid_flat = valid_mask_flat.sum().clamp(min=1.0)
            if hasattr(self.vecenv.driver_env, "entropy_conditioned") and self.vecenv.driver_env.entropy_conditioned:
                mb_obs_flat = mb_obs.reshape(-1, mb_obs.shape[-1])

                if (
                    hasattr(self.vecenv.driver_env, "dynamics_model")
                    and self.vecenv.driver_env.dynamics_model == "jerk"
                ):
                    ent_idx = 12  # EGO_FEATURES_JERK (was 10 before lane features)
                else:
                    ent_idx = 9  # EGO_FEATURES_CLASSIC (was 7 before lane features)

                if self.vecenv.driver_env.reward_conditioned:
                    ent_idx += 3

                ent_weights = mb_obs_flat[:, ent_idx]  # after ego(7/10) + RC(3)
                ent_weights = ent_weights.reshape(entropy.shape)
                entropy_loss = -((entropy * ent_weights) * valid_mask_flat).sum() / n_valid_flat
                loss = pg_loss + config["vf_coef"] * v_loss + entropy_loss
            else:
                entropy_loss = (entropy * valid_mask_flat).sum() / n_valid_flat
                loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss
            self.amp_context.__enter__()  # TODO: AMP needs some debugging

            # Write back the new value-head output for the next outer GAE.
            # CRITICAL: preserve limbo positions — at those slots `newvalue`
            # was computed from garbage obs (INVALID_POSITION) and writing
            # it back would poison subsequent GAE calls. The old `mb_values`
            # at limbo positions is also garbage (also computed from limbo
            # obs at rollout time), so neither choice is "right" — but
            # keeping the prior value at limbo positions prevents
            # mb-by-mb drift across PPO epochs.
            new_v = newvalue.detach().float()
            if mb_removed is not None:
                new_v = torch.where(mb_removed, mb_values.float(), new_v)
            self.values[idx] = new_v

            # Logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item() / self.total_minibatches
            losses["value_loss"] += v_loss.item() / self.total_minibatches
            losses["entropy"] += entropy_loss.item() / self.total_minibatches
            losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
            losses["approx_kl"] += approx_kl.item() / self.total_minibatches
            losses["clipfrac"] += clipfrac.item() / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses["explained_variance"] = explained_var.item()

        profile.end()
        logs = None
        if _TRIAL_DEBUG_ENABLED:
            # Per-epoch summary: cache health, transformer position, loss snapshot.
            k_cache = getattr(self, "transformer_k_cache", None)
            v_cache = getattr(self, "transformer_v_cache", None)
            pos_buf = getattr(self, "transformer_position", None)
            cache_stats = {}
            if k_cache is not None and isinstance(k_cache, dict) and len(k_cache) > 0:
                # k_cache is dict keyed by transformer_key; each value is a list of layer K tensors
                some_key = next(iter(k_cache.keys()))
                cache_list = k_cache[some_key]
                if cache_list is not None and len(cache_list) > 0:
                    sample = cache_list[0]
                    cache_stats = dict(
                        shape=list(sample.shape),
                        dtype=str(sample.dtype),
                        norm_mean=float(sample.norm(dim=-1).mean().item()),
                        nan_count=int(torch.isnan(sample).sum().item()),
                        inf_count=int(torch.isinf(sample).sum().item()),
                    )
            pos_stats = {}
            if pos_buf is not None and isinstance(pos_buf, dict) and len(pos_buf) > 0:
                some_key = next(iter(pos_buf.keys()))
                p = pos_buf[some_key]
                if p is not None:
                    pos_stats = dict(min=int(p.min().item()), max=int(p.max().item()))
            _trial_debug_log(
                "epoch_end",
                epoch=int(self.epoch),
                step=int(self.global_step),
                policy_loss=float(losses.get("policy_loss", 0)),
                value_loss=float(losses.get("value_loss", 0)),
                entropy=float(losses.get("entropy", 0)),
                approx_kl=float(losses.get("approx_kl", 0)),
                clipfrac=float(losses.get("clipfrac", 0)),
                explained_var=float(explained_var.item() if not torch.isnan(torch.tensor(float(explained_var))) else 0.0),
                cache=cache_stats,
                position=pos_stats,
            )
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

            if self.render and self.epoch % self.render_interval == 0:
                torch.cuda.empty_cache()
                pufferlib.utils.render_videos(
                    config=self.config,
                    policy=self.uncompiled_policy,
                    logger=self.logger,
                    epoch=self.epoch,
                    global_step=self.global_step,
                    device=self.config["device"],
                )

        if self.config["eval"]["wosac_realism_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            pufferlib.utils.run_wosac_eval_in_subprocess(self.config, self.logger, self.global_step)

        if self.config["eval"]["human_replay_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            pufferlib.utils.run_human_replay_eval_in_subprocess(self.config, self.logger, self.global_step)
            torch.cuda.empty_cache()
            pufferlib.utils.render_videos(
                config=self.config,
                policy=self.uncompiled_policy,
                logger=self.logger,
                epoch=self.epoch,
                global_step=self.global_step,
                device=self.config["device"],
                human_replay=True,
            )

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config["device"]
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            "SPS": dist_sum(self.sps, device),
            "agent_steps": agent_steps,
            "uptime": time.time() - self.start_time,
            "epoch": int(dist_sum(self.epoch, device)),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{f"environment/{k}": v for k, v in self.stats.items()},
            **{f"losses/{k}": v for k, v in self.losses.items()},
            **{f"performance/{k}": v["elapsed"] for k, v in self.profile},
            # **{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            # **{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            # **{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                self.logger.log(logs, agent_steps)
                return logs
            else:
                return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}.pt")
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}")
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{self.config['env']}_{self.epoch:06d}.pt"
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "run_id": run_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)

        # Sidecar metadata: every render/eval can recover the right
        # conditioning, dataset, and architecture from <run_dir>/info.json
        # without the user having to re-pass them on the CLI.
        write_run_info(path, self.config, run_id)

        return model_path

    def print_dashboard(self, clear=False, idx=[0], c1="[cyan]", c2="[white]", b1="[bright_cyan]", b2="[bright_white]"):
        config = self.config
        sps = dist_sum(self.sps, config["device"])
        agent_steps = dist_sum(self.global_step, config["device"])
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True, show_header=False, border_style="bright_cyan")
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f"{b1}PufferLib {b2}3.0 {idx[0] * ' '}:blowfish:",
            f"{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%",
            f"{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%",
            f"{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%",
            f"{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%",
        )
        idx[0] = (idx[0] - 1) % 10

        s = Table(box=None, expand=True)
        remaining = "A hair past a freckle"
        if sps != 0:
            remaining = duration((config["total_timesteps"] - agent_steps) / sps, b2, c2)

        s.add_column(f"{c1}Summary", justify="left", vertical="top", width=10)
        s.add_column(f"{c1}Value", justify="right", vertical="top", width=14)
        s.add_row(f"{c2}Env", f"{b2}{config['env']}")
        s.add_row(f"{c2}Params", abbreviate(self.model_size, b2, c2))
        s.add_row(f"{c2}Steps", abbreviate(agent_steps, b2, c2))
        s.add_row(f"{c2}SPS", abbreviate(sps, b2, c2))
        s.add_row(f"{c2}Epoch", f"{b2}{self.epoch}")
        s.add_row(f"{c2}Uptime", duration(self.uptime, b2, c2))
        s.add_row(f"{c2}Remaining", remaining)

        delta = profile.eval["buffer"] + profile.train["buffer"]
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf("Evaluate", b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf("  Env", c2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf("Train", b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf("  Learn", c2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.train_misc, b2, c2))

        l = Table(
            box=None,
            expand=True,
        )
        l.add_column(f"{c1}Losses", justify="left", width=16)
        l.add_column(f"{c1}Value", justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try:  # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print("\033[0;0H" + capture.get())


def compute_puff_advantage(
    values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
):
    """CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version."""

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(
        values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
    )

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return str(num)
    elif num < 1e6:
        return f"{num / 1e3:.1f}K"
    elif num < 1e9:
        return f"{num / 1e6:.1f}M"
    elif num < 1e12:
        return f"{num / 1e9:.1f}B"
    else:
        return f"{num / 1e12:.2f}T"


def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"


def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100 * prof["buffer"] / delta_ref - 1e-5)
    return f"{color}{name}", duration(prof["elapsed"], b2, c2), f"{b2}{percent:2d}{c2}%"


def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()


class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        if epoch % self.frequency != 0:
            return

        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]["start"] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile["start"]
        profile["elapsed"] += delta
        profile["delta"] += delta

    def end(self):
        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof["delta"] > 0:
                prof["buffer"] = prof["delta"]
                prof["delta"] = 0


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100 * psutil.cpu_percent() / psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                    time.sleep(self.delay)
                    continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * (total - free) / total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def downsample(arr, m):
    if len(arr) < m:
        return arr

    if m == 0:
        return [arr[-1]]

    orig_arr = arr
    last = arr[-1]
    arr = arr[:-1]
    arr = np.array(arr)
    n = len(arr)
    n = (n // m) * m
    arr = arr[-n:]
    downsampled = arr.reshape(m, -1).mean(axis=1)
    return np.concatenate([downsampled, [last]])


class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100 * time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
        pass


class NeptuneLogger:
    def __init__(self, args, load_id=None, mode="async"):
        import neptune as nept

        neptune_name = args["neptune_name"]
        neptune_project = args["neptune_project"]
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags=[args["tag"]] if args["tag"] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def close(self, model_path):
        self.neptune["model"].track_files(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination="artifacts")
        return f"artifacts/{self.run_id}.pt"


class WandbLogger:
    def __init__(self, args, load_id=None, resume="allow"):
        import wandb

        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args["wandb_project"],
            group=args["wandb_group"],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            name=args.get("wandb_name"),
            tags=[args["tag"]] if args["tag"] is not None else [],
        )
        self.wandb = wandb
        self.run_id = wandb.run.id

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def close(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f"{self.run_id}:latest")
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f"{data_dir}/{model_file}"


def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print("World size", world_size)
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)
        if hasattr(policy, "lstm"):
            # model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args["neptune"]:
        logger = NeptuneLogger(args, load_id=args.get("load_id"))
    elif args["wandb"]:
        # Pass load_id so the wandb logger resumes the existing run instead
        # of creating a fresh one. WandbLogger uses resume="allow", so wandb
        # picks up where the original run left off (history, name, tags).
        logger = WandbLogger(args, load_id=args.get("load_id"))

    train_config = dict(
        **args["train"],
        env=env_name,
        eval=args.get("eval", {}),
        env_config=args.get("env", {}),
        policy_architecture=args.get("policy_architecture", "Recurrent"),
        # rnn_name lives at args top level — must be explicitly propagated,
        # else config.get("rnn_name") defaults to "Recurrent" and the
        # Transformer init/rollout branches in PuffeRL never fire.
        rnn_name=args.get("rnn_name", args.get("policy_architecture", "Recurrent")),
        load_model_path=args.get("load_model_path"),
        load_id=args.get("load_id"),
    )
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    all_logs = []
    while pufferl.global_step < train_config["total_timesteps"]:
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20 * train_config["total_timesteps"]:
                all_logs.append(logs)

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    i = 0
    stats = {}
    while i < 32 or not stats:
        stats = pufferl.evaluate()
        i += 1

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs


def eval(env_name, args=None, vecenv=None, policy=None):
    """Evaluate a policy."""

    args = args or load_config(env_name)

    wosac_enabled = args["eval"]["wosac_realism_eval"]
    human_replay_enabled = args["eval"]["human_replay_eval"]
    # Honor eval.map_dir only when explicitly set; otherwise inherit the
    # training env.map_dir so eval doesn't silently switch datasets.
    eval_map_dir = args["eval"].get("map_dir")
    if eval_map_dir in (None, "", "None"):
        eval_map_dir = args["env"].get("map_dir")
    args["env"]["map_dir"] = eval_map_dir
    args["eval"]["map_dir"] = eval_map_dir
    args["env"]["num_maps"] = args["eval"]["num_maps"]
    args["env"]["use_all_maps"] = True
    dataset_name = args["env"]["map_dir"].split("/")[-1]

    if wosac_enabled:
        print(f"Running WOSAC realism evaluation with {dataset_name} dataset. \n")
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

        backend = args["eval"]["backend"]
        assert backend == "PufferEnv" or not wosac_enabled, "WOSAC evaluation only supports PufferEnv backend."
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
        args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
        args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
        args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
        args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        evaluator = WOSACEvaluator(args)

        # Collect ground truth trajectories from the dataset
        gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

        print(f"Number of scenarios: {len(np.unique(gt_trajectories['scenario_id']))}")
        print(f"Number of controlled agents: {gt_trajectories['x'].shape[0]}")
        print(f"Number of evaluated agents: {np.sum(gt_trajectories['id'] >= 0)}")

        # Roll out trained policy in the simulator
        simulated_trajectories = evaluator.collect_simulated_trajectories(args, vecenv, policy)

        if args["eval"]["wosac_sanity_check"]:
            evaluator._quick_sanity_check(gt_trajectories, simulated_trajectories)

        # Analyze and compute metrics
        agent_state = vecenv.driver_env.get_global_agent_state()
        road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
        results = evaluator.compute_metrics(
            gt_trajectories,
            simulated_trajectories,
            agent_state,
            road_edge_polylines,
            args["eval"]["wosac_aggregate_results"],
        )

        if args["eval"]["wosac_aggregate_results"]:
            import json

            print("\nWOSAC_METRICS_START")
            print(json.dumps(results))
            print("WOSAC_METRICS_END")

        return results

    elif human_replay_enabled:
        print(f"Running human replay evaluation with {dataset_name} dataset.\n")
        from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

        backend = args["eval"].get("backend", "PufferEnv")
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
        # episode_length is NOT hardcoded here — inherits scenario_length
        # from training config (WOMD=91, nuPlan=201).
        # Human replay: only 1 ego is policy-controlled, others follow logged trajectories
        args["env"]["co_player_enabled"] = False
        args["env"]["max_controlled_agents"] = 1
        # `human_replay_mode` is only accepted by AdaptiveDrivingAgent
        if "adaptive" in env_name:
            args["env"]["human_replay_mode"] = True
        if args["eval"].get("human_replay_num_agents") is not None:
            args["env"]["num_agents"] = args["eval"]["human_replay_num_agents"]
        if args["eval"].get("human_replay_num_maps") is not None:
            args["env"]["num_maps"] = args["eval"]["human_replay_num_maps"]
        if args["eval"].get("map_dir") not in (None, "", "None"):
            args["env"]["map_dir"] = args["eval"]["map_dir"]

        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        print(f"Effective number of scenarios used: {len(vecenv.driver_env.agent_offsets) - 1}")

        evaluator = HumanReplayEvaluator(args)

        # Run rollouts with human replays
        results = evaluator.rollout(args, vecenv, policy)

        import json

        print("HUMAN_REPLAY_METRICS_START")
        print(json.dumps(results))
        print("HUMAN_REPLAY_METRICS_END")

        return results
    else:  # Standard evaluation: Render
        backend = args["vec"]["backend"]
        if backend != "PufferEnv":
            backend = "Serial"

        args["vec"] = dict(backend=backend, num_envs=1)
        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        ob, info = vecenv.reset()
        driver = vecenv.driver_env
        num_agents = vecenv.observation_space.shape[0]
        device = args["train"]["device"]

        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        frames = []
        while True:
            render = driver.render()
            if len(frames) < args["save_frames"]:
                frames.append(render)

            # Screenshot Ocean envs with F12, gifs with control + F12
            if driver.render_mode == "ansi":
                print("\033[0;0H" + render + "\n")
                time.sleep(1 / args["fps"])
            elif driver.render_mode == "rgb_array":
                pass
                # import cv2
                # render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                # cv2.imshow('frame', render)
                # cv2.waitKey(1)
                # time.sleep(1/args['fps'])

            with torch.no_grad():
                ob = torch.as_tensor(ob).to(device)
                logits, value = policy.forward_eval(ob, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            ob = vecenv.step(action)[0]

            if len(frames) > 0 and len(frames) == args["save_frames"]:
                import imageio

                imageio.mimsave(args["gif_path"], frames, fps=args["fps"], loop=0)
                frames.append("Done")


def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Sweeps require either wandb or neptune")

    method = args["sweep"].pop("method")
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f"Invalid sweep method {method}. See pufferlib.sweep")

    sweep = sweep_cls(args["sweep"])
    points_per_run = args["sweep"]["downsample"]
    target_key = f"environment/{args['sweep']['metric']}"
    for i in range(args["max_runs"]):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sweep.suggest(args)
        total_timesteps = args["train"]["total_timesteps"]
        all_logs = train(env_name, args=args)
        all_logs = [e for e in all_logs if target_key in e]
        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log["uptime"] for log in all_logs], points_per_run)
        timesteps = downsample([log["agent_steps"] for log in all_logs], points_per_run)
        for score, cost, timestep in zip(scores, costs, timesteps):
            args["train"]["total_timesteps"] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args["train"]["total_timesteps"] = total_timesteps


def controlled_exp(env_name, args=None):
    """Run experiments with all combinations of specified parameter values."""
    import itertools
    from copy import deepcopy

    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Targeted experiments require either wandb or neptune")

    # Check if controlled_exp config exists
    if "controlled_exp" not in args:
        raise pufferlib.APIUsageError("No [controlled_exp.*] sections found in config")

    # Extract parameters from controlled_exp namespace
    params = {}
    for section, section_config in args["controlled_exp"].items():
        if isinstance(section_config, dict):
            for param, param_config in section_config.items():
                if isinstance(param_config, dict) and "values" in param_config:
                    params[f"{section}.{param}"] = param_config["values"]

    if not params:
        raise pufferlib.APIUsageError("No parameters with 'values' lists found in [controlled_exp.*] sections")

    # Generate all combinations
    keys = list(params.keys())
    combinations = list(itertools.product(*[params[k] for k in keys]))

    print(f"Running a total of {len(combinations)} experiments with parameters: {keys}")

    # Run each combination
    for i, combo in enumerate(combinations, 1):
        exp_args = deepcopy(args)

        # Set parameters
        for key, value in zip(keys, combo):
            section, param = key.split(".")
            exp_args[section][param] = value

        print(f"\nExperiment {i}/{len(combinations)}: {dict(zip(keys, combo))}")

        # Train
        train(env_name, args=exp_args)

    print(f"\n✓ Completed all {len(combinations)} experiments")


def sanity(env_name, args=None):
    args = args or load_config(env_name)
    base_dir = Path(__file__).resolve().parent / "resources" / "drive" / "sanity"
    json_dir = base_dir / "sanity_jsons"
    binary_dir = base_dir / "sanity_binaries"

    available_maps = {p.stem: p for p in json_dir.glob("*.json")}
    selected = args.get("sanity_maps")
    if isinstance(selected, str):
        selected = [selected]

    if selected:
        missing = [name for name in selected if name not in available_maps]
        if missing:
            raise pufferlib.APIUsageError(f"Unknown sanity maps: {', '.join(sorted(missing))}")
        chosen = [(name, available_maps[name]) for name in selected]
    else:
        chosen = sorted(available_maps.items())

    if not chosen:
        raise pufferlib.APIUsageError(f"No sanity maps found in {json_dir}")

    from pufferlib.ocean.drive.drive import load_map

    binary_dir.mkdir(parents=True, exist_ok=True)
    binaries = []
    for idx, (name, json_path) in enumerate(chosen):
        output_path = binary_dir / f"{name}.bin"
        load_map(str(json_path), idx, str(output_path))
        binaries.append((name, output_path))

    runs = []
    for name, binary in binaries:
        map_zero = binary_dir / "map_000.bin"
        shutil.copy2(binary, map_zero)

        run_args = {
            **args,
            "env": {**args["env"], "num_maps": 1, "map_dir": str(binary_dir)},
            "train": {**args["train"], "render_map": str(map_zero)},
        }
        if run_args.get("wandb"):
            run_args["wandb_name"] = name

        print(f"Running sanity map '{name}' from {binary.name}")
        run_logs = train(env_name=env_name, args=run_args)
        runs.append({"map": name, "logs": run_logs})

    print("Sanity checklist:")
    for entry in runs:
        name = entry["map"]
        logs = entry.get("logs") or []
        final = logs[-1] if logs else {}
        score = final.get("environment/score")
        if score is None:
            status = "unknown (no score)"
        elif score >= 0.95:
            status = "✅ Solved"
        else:
            status = "❌ unsolved"
        print(f" - {name}: {status} (score={score})")

    return runs


def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args["train"], env=args["env_name"], tag=args["tag"])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args["neptune"], wandb=args["wandb"])

    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")


def export(args=None, env_name=None, vecenv=None, policy=None, path=None, silent=False):
    args = args or load_config(env_name)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)
    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        if not silent:
            print(name, param.shape, param.data.cpu().numpy().ravel()[0])

    weights = np.concatenate(weights)
    if path is None:
        path = f"pufferlib/resources/drive/{args['env_name']}_weights.bin"

    weights.tofile(path)

    if not silent:
        print(f"Saved {len(weights)} weights to {path}")


def write_run_info(run_dir, config, run_id):
    """Persist the bits of training config that render/eval need to replay.

    We only record fields that change observation/architecture shape or
    dataset identity — the things you can't recover from the .pt alone.
    Existing checkpoints don't have this file; readers must treat it as
    optional.
    """
    import json

    env_cfg = config.get("env_config", {})
    info = {
        "run_id": run_id,
        "env_name": config.get("env"),
        "policy_architecture": config.get("policy_architecture"),
        "rnn_name": config.get("rnn_name"),
        "env": {
            "map_dir": env_cfg.get("map_dir"),
            "num_maps": env_cfg.get("num_maps"),
            "num_agents": env_cfg.get("num_agents"),
            "num_ego_agents": env_cfg.get("num_ego_agents"),
            "k_scenarios": env_cfg.get("k_scenarios", 1),
            "scenario_length": env_cfg.get("scenario_length", 91),
            "dynamics_model": env_cfg.get("dynamics_model", "classic"),
            "co_player_enabled": bool(env_cfg.get("co_player_enabled")),
            "conditioning": env_cfg.get("conditioning", {}),
            "co_player_policy": env_cfg.get("co_player_policy", {}),
        },
    }
    info_path = os.path.join(run_dir, "info.json")
    try:
        with open(info_path + ".tmp", "w") as f:
            json.dump(info, f, indent=2, default=str)
        os.rename(info_path + ".tmp", info_path)
    except Exception as e:
        print(f"[info.json] failed to write: {e}")


def load_run_info(model_path):
    """Look up the run_dir/info.json for a given checkpoint path. Returns {} if missing."""
    import json

    candidates = []
    parent = os.path.dirname(model_path) or "."
    candidates.append(os.path.join(parent, "info.json"))
    # Allow `experiments/puffer_drive_<id>.pt` (flat copy) → look in sibling run dir.
    base, ext = os.path.splitext(model_path)
    if ext == ".pt" and os.path.basename(base).startswith("puffer_"):
        candidates.append(os.path.join(base, "info.json"))
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"[info.json] failed to read {path}: {e}")
    return {}


def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    env_name = args["env_name"]
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args["train"]["env_batch_size"])


def load_env(env_name, args):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args["env"], **args["vec"])


def load_policy(args, vecenv, env_name=""):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)

    device = args["train"]["device"]

    load_id = args["load_id"]
    load_path = args.get("load_model_path")
    state_dict = None
    rnn_name = args.get("policy_architecture", "Recurrent")

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    elif load_id is not None:
        if args["neptune"]:
            path = NeptuneLogger(args, load_id, mode="read-only").download()
        elif args["wandb"]:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError("No run id provided for eval")
        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Auto-detect architecture from state_dict keys
    if state_dict is not None:
        if "positional_embedding" in state_dict:
            rnn_name = "Transformer"
        elif "lstm.weight_ih_l0" in state_dict:
            rnn_name = "Recurrent"

    policy_cls = getattr(env_module.torch, args["policy_name"])
    policy = policy_cls(vecenv.driver_env, **args["policy"])

    # Handle both RNN and Transformer wrappers via rnn_name
    if rnn_name == "Transformer":
        # Load transformer wrapper
        transformer_cls = getattr(env_module.torch, rnn_name)
        # For adaptive_driving_agent, use episode_length as horizon (k_scenarios * scenario_length)
        # Otherwise, use config horizon with fallback to episode_length
        is_adaptive = getattr(vecenv.driver_env, "env_name", None) == "adaptive_drive"
        if is_adaptive:
            args["transformer"]["horizon"] = vecenv.driver_env.episode_length
        else:
            args["transformer"]["horizon"] = args["train"].get("horizon", vecenv.driver_env.episode_length)
        policy = transformer_cls(vecenv.driver_env, policy, **args["transformer"])
    elif rnn_name is not None:
        # Load RNN wrapper (Recurrent)
        rnn_cls = getattr(env_module.torch, rnn_name)
        policy = rnn_cls(vecenv.driver_env, policy, **args["rnn"])

    policy = policy.to(device)

    # Load the state dict if we have one
    if state_dict is not None:
        policy.load_state_dict(state_dict)

    return policy


def load_config(env_name):
    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]"
        " demo options. Shows valid args for your env and policy",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to a pretrained checkpoint")
    parser.add_argument(
        "--load-id", type=str, default=None, help="Kickstart/eval from from a finished Wandb/Neptune run"
    )
    parser.add_argument(
        "--render-mode", type=str, default="auto", choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"]
    )
    parser.add_argument("--save-frames", type=int, default=0)
    parser.add_argument("--gif-path", type=str, default="eval.gif")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-runs", type=int, default=200, help="Max number of sweep runs")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging", default=True)
    parser.add_argument("--wandb-project", type=str, default="ada")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument("--neptune", action="store_true", help="Use neptune for logging")
    parser.add_argument("--neptune-name", type=str, default="pufferai")
    parser.add_argument("--neptune-project", type=str, default="ablations")
    parser.add_argument("--local-rank", type=int, default=0, help="Used by torchrun for DDP")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    parser.add_argument("--sanity-maps", nargs="*", default=None, help="Optional list of sanity map base names to run")
    args = parser.parse_known_args()[0]

    # Load defaults and config
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, "config/**/*.ini")
    puffer_default_config = os.path.join(puffer_dir, "config/default.ini")
    if env_name == "default":
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p["base"]["env_name"].split():
                break
        else:
            raise pufferlib.APIUsageError("No config for env_name {}".format(env_name))

    # Dynamic help menu from config
    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    for section in p.sections():
        for key in p[section]:
            fmt = f"--{key}" if section == "base" else f"--{section}.{key}"
            parser.add_argument(fmt.replace("_", "-"), default=puffer_type(p[section][key]), type=puffer_type)

    parser.add_argument(
        "-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit"
    )

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args["train"]["use_rnn"] = args["rnn_name"] is not None
    return args


def main():
    err = "Usage: puffer [train, eval, sweep, controlled_exp, autotune, profile, export, sanity] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "sweep":
        sweep(env_name=env_name)
    elif mode == "controlled_exp":
        controlled_exp(env_name=env_name)
    elif mode == "autotune":
        autotune(env_name=env_name)
    elif mode == "profile":
        profile(env_name=env_name)
    elif mode == "export":
        export(env_name=env_name)
    elif mode == "sanity":
        sanity(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    main()
