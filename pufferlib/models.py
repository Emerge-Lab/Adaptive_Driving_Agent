import os

import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces

import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


# Set PUFFER_TRANSFORMER_LEGACY_EVAL=1 to fall back to the pre-KV-cache path.
_USE_LEGACY_EVAL = os.environ.get("PUFFER_TRANSFORMER_LEGACY_EVAL", "0") == "1"


class Default(nn.Module):
    """Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    """

    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space, pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict)
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict)

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            num_obs = np.prod(env.single_observation_space.shape)
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
                nn.GELU(),
            )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, num_atns), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
            )
            self.decoder_logstd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        """Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers)."""
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else:
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        """Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers)."""
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values


class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        """Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example."""
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if "layer_norm" in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        # self.pre_layernorm = nn.LayerNorm(hidden_size)
        # self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward_eval(self, observations, state):
        """Forward function for inference. 3x faster than using LSTM directly"""
        hidden = self.policy.encode_observations(observations, state=state)
        h = state["lstm_h"]
        c = state["lstm_c"]

        # TODO: Don't break compile
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], "LSTM state must be (h, c)"
            lstm_state = (h, c)
        else:
            lstm_state = None

        # hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        # hidden = self.post_layernorm(hidden)
        state["hidden"] = hidden
        state["lstm_h"] = hidden
        state["lstm_c"] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        """Forward function for training. Uses LSTM for fast time-batching"""
        x = observations
        lstm_h = state["lstm_h"]
        lstm_c = state["lstm_c"]

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B, "LSTM state must be (h, c)"
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B * TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B * TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        # hidden = self.pre_layernorm(hidden)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()

        # hidden = self.post_layernorm(hidden)
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B * TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        # state.batch_logits = logits.reshape(B, TT, -1)
        state["hidden"] = hidden
        state["lstm_h"] = lstm_h.detach()
        state["lstm_c"] = lstm_c.detach()
        return logits, values


class TransformerWrapper(nn.Module):  # TransformerWrapper
    def __init__(
        self,
        env,
        policy,
        input_size=128,
        hidden_size=128,
        num_layers=4,
        num_heads=8,
        horizon=512,
        dropout=0.0,
        use_checkpointing=False,
    ):
        """Wraps your policy with a Transformer for temporal modeling.

        Args:
            env: Environment instance
            policy: Your Drive policy (must have encode_observations and decode_actions)
            input_size: Size of encoded observations (from policy.encode_observations)
            hidden_size: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            horizon: Maximum sequence length to attend over
            dropout: Dropout probability
            use_checkpointing: Enable gradient checkpointing to save memory (slower training)
        """
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.num_layers = num_layers
        self.num_heads = num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_size // num_heads
        self.is_continuous = self.policy.is_continuous
        self.use_checkpointing = use_checkpointing
        # Per-slot attention masks for KV-cached streaming inference. Cached
        # lazily per device to avoid recomputing the same mask each step.
        self._streaming_mask_cache = {}

        # Project encoded observations to transformer dimension if needed
        if input_size != hidden_size:
            self.input_projection = nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = nn.Identity()

        # Sinusoidal positional embeddings (Vaswani et al.) — non-trainable.
        # Per-episode reset is applied in forward() (training) so the PE
        # indexing matches forward_eval's cache-pos indexing under
        # multi-episode-per-row rollouts.
        pe = torch.zeros(horizon, hidden_size)
        position = torch.arange(0, horizon, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_embedding", pe.unsqueeze(0))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # create cache for memory context
        for T in [1, 2, 4, 8, 16, 32, 64, 91, 182, 273, 364, 455]:
            mask = self.create_causal_mask(T, "cpu")
            self.register_buffer(f"_causal_mask_{T}", mask, persistent=False)

        # Cached masks for episode mask creation (reduces memory allocation)
        self.register_buffer("_zero_mask", torch.zeros(1), persistent=False)
        self.register_buffer("_neg_inf_mask", torch.full((1,), float("-inf")), persistent=False)

        # Layer norm for output
        self.output_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to GPT-2"""
        for name, param in self.named_parameters():
            if "layer_norm" in name or "layernorm" in name or "output_norm" in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

    def create_causal_mask(self, seq_len, device):
        """Create causal attention mask"""
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask

    def get_causal_mask(self, T, device):
        """Get cached causal mask or create new one"""
        buffer_name = f"_causal_mask_{T}"
        if hasattr(self, buffer_name):
            mask = getattr(self, buffer_name)
            if mask.device != device:
                # Move to device and cache
                mask = mask.to(device)
                setattr(self, buffer_name, mask)
            return mask
        mask = self.create_causal_mask(T, device)
        self.register_buffer(buffer_name, mask, persistent=False)
        return mask

    def get_positional_embedding(self, T, device):
        """Get cached positional embedding for length T."""
        cache_key = f"_pos_embed_{T}"
        if not hasattr(self, cache_key) or getattr(self, cache_key).device != device:
            pos_embed = self.positional_embedding[:, :T].to(device)
            setattr(self, cache_key, pos_embed)
        return getattr(self, cache_key)

    @staticmethod
    def compute_pos_within_episode(terminals):
        """For terminals (B, T) bool/float, return per-slot position within
        its episode (resets at slot AFTER each terminal). The convention
        matches create_episode_mask: the terminal slot itself belongs to
        the OLD episode, and the new episode starts at slot terminal+1.

        Vectorized: shift terminals right by one (so a terminal at slot s
        becomes a start-flag at slot s+1), multiply by arange to mark the
        position of each episode-start, cummax to propagate the last
        seen start position forward, then subtract from arange.
        """
        B, T = terminals.shape
        device = terminals.device
        arange_T = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)
        shifted = F.pad(terminals[:, :-1], (1, 0)).long()  # (B, T)
        starts = arange_T * shifted  # (B, T) — slot index where new episode begins (else 0)
        ep_start = starts.cummax(dim=1).values  # (B, T) — most recent episode-start at or before t
        return arange_T - ep_start  # (B, T)

    def create_episode_mask(self, terminals, seq_len):
        """Episode mask which ensures that you arent attending over episode boundaries.
        Optimized with cached mask buffers to reduce memory allocation."""
        B = terminals.shape[0]
        device = terminals.device

        # Use cumsum for episode IDs
        episode_ids = torch.nn.functional.pad(terminals[:, :-1], (1, 0)).cumsum(dim=1)

        # Avoid full (B, T, T) allocation - use sparse comparison
        mask_allow = episode_ids.unsqueeze(2) == episode_ids.unsqueeze(1)

        # Use cached tensors moved to correct device
        zero_mask = self._zero_mask.to(device) if self._zero_mask.device != device else self._zero_mask
        neg_inf_mask = self._neg_inf_mask.to(device) if self._neg_inf_mask.device != device else self._neg_inf_mask

        return torch.where(mask_allow, zero_mask, neg_inf_mask)

    # ------------------------------------------------------------------ #
    # Streaming inference with per-layer KV cache.
    #
    # The legacy `_forward_eval_legacy` below recomputes a full transformer
    # forward over the entire horizon-length context buffer on every step,
    # then throws away all but one output position. For B=512 co-players,
    # horizon=91, single-thread CPU, this costs ~3 s per step and dominates
    # training wallclock.
    #
    # The KV-cached path maintains per-layer (K, V) buffers in `state` and
    # only computes Q/K/V for the new token, attending against the cache.
    # Numerically equivalent to the legacy path (same rolling buffer +
    # causal-row semantics, including the post-wrap "self-attention only"
    # behavior at slot 0 after horizon steps).
    #
    # Implementation note: `slot` is kept as a 1-element long tensor (not
    # `int(pos.item())`) so this method stays compile-friendly. On the GPU
    # ego policy under torch.compile, `.item()` would cause a Dynamo graph
    # break and a CUDA sync every call.
    # ------------------------------------------------------------------ #

    def _slot_arange(self, device):
        """Return a length-`horizon` arange tensor on `device`, cached per device."""
        key = (device.type, device.index if device.index is not None else -1)
        cached = self._streaming_mask_cache.get(key)
        if cached is not None:
            return cached
        arr = torch.arange(self.horizon, device=device)
        self._streaming_mask_cache[key] = arr
        return arr

    def _make_kv_cache(self, batch_size, device, dtype):
        return [
            torch.zeros(batch_size, self.num_heads, self.horizon, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

    def init_eval_state(self, batch_size, device, dtype=torch.float32):
        """Allocate a fresh streaming-inference state dict for this policy."""
        return dict(
            k_cache=self._make_kv_cache(batch_size, device, dtype),
            v_cache=self._make_kv_cache(batch_size, device, dtype),
            transformer_position=torch.zeros(1, dtype=torch.long, device=device),
        )

    def _prime_kv_cache(self, indices, state):
        """Prime K/V cache for `indices` to match legacy 'zero hidden context'.

        The legacy reset only zeroed the rolling hidden buffer. Because that
        buffer is summed with the slot-tied positional embedding inside the
        transformer, the *effective* K/V at unwritten slots is the K/V you
        get from running the transformer over an all-zero hidden sequence
        (i.e. just the pos embeddings, with causal attention propagating
        through layers). This priming fills our cache with exactly that
        state, so subsequent forward_eval calls match the legacy bit-close.
        """
        if isinstance(indices, slice) and indices == slice(None):
            n_idx = state["k_cache"][0].shape[0]
        elif torch.is_tensor(indices):
            n_idx = int(indices.shape[0])
        else:
            n_idx = len(indices)
        if n_idx == 0:
            return

        H, D = self.num_heads, self.head_dim
        T = self.horizon
        device = state["k_cache"][0].device
        dtype = state["k_cache"][0].dtype

        pos_embed = self.get_positional_embedding(T, device).to(dtype)  # (1, T, hidden)
        layer_input = pos_embed.expand(n_idx, T, self.hidden_size).contiguous()
        causal_mask = self.get_causal_mask(T, device)

        with torch.no_grad():
            for li, layer in enumerate(self.transformer.layers):
                attn = layer.self_attn
                x_norm = layer.norm1(layer_input)
                qkv = F.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(n_idx, T, H, D).transpose(1, 2)
                k = k.view(n_idx, T, H, D).transpose(1, 2)
                v = v.view(n_idx, T, H, D).transpose(1, 2)

                # Defensive cast: under autocast or mixed-precision, k/v can
                # come out in a different dtype than the cache; PyTorch refuses
                # cross-dtype `index_put`. Match the cache's dtype.
                state["k_cache"][li][indices] = k.to(state["k_cache"][li].dtype)
                state["v_cache"][li][indices] = v.to(state["v_cache"][li].dtype)

                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)
                attn_out = attn_out.transpose(1, 2).reshape(n_idx, T, self.hidden_size)
                attn_out = F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)
                x = layer_input + attn_out
                x_norm2 = layer.norm2(x)
                ffn_h = layer.activation(F.linear(x_norm2, layer.linear1.weight, layer.linear1.bias))
                ffn_out = F.linear(ffn_h, layer.linear2.weight, layer.linear2.bias)
                layer_input = x + ffn_out

    def reset_eval_state(self, state, done_indices=None):
        """Reset KV cache (and step counter) for done agents.

        - done_indices=None: full reset. K/V zeroed, step counter cleared.
          (Equivalent to allocating a fresh state.)
        - done_indices=tensor/array of agent indices: re-prime those rows
          to mirror the legacy "zero hidden buffer" behavior. The shared
          step counter is intentionally NOT reset in that case (matches
          legacy, which only touched per-row context).
        """
        k_cache = state.get("k_cache")
        v_cache = state.get("v_cache")
        if k_cache is None or v_cache is None:
            return
        if done_indices is None:
            for c in k_cache:
                c.zero_()
            for c in v_cache:
                c.zero_()
            pos = state.get("transformer_position")
            if pos is not None:
                pos.zero_()
        else:
            idx = done_indices
            if not torch.is_tensor(idx):
                idx = torch.as_tensor(idx, device=k_cache[0].device, dtype=torch.long)
            self._prime_kv_cache(idx, state)

    def forward_eval(self, observations, state):
        if _USE_LEGACY_EVAL:
            # Escape hatch for benchmarking / safety net: set
            # PUFFER_TRANSFORMER_LEGACY_EVAL=1 in the environment to bypass
            # the KV-cached path and use the original full-context forward.
            return self._forward_eval_legacy(observations, state)
        B = observations.shape[0]
        device = observations.device

        hidden = self.policy.encode_observations(observations, state=state)
        hidden = self.input_projection(hidden)
        # hidden: (B, hidden_size)

        # Fetch or lazily allocate the KV cache. We re-allocate if shape
        # changes (e.g. batch size differs across calls) or dtype mismatches
        # the input (mixed-precision boundary).
        k_cache = state.get("k_cache")
        v_cache = state.get("v_cache")
        need_alloc = (
            k_cache is None or v_cache is None or k_cache[0].shape[0] != B or k_cache[0].shape[2] != self.horizon
        )
        if need_alloc:
            k_cache = self._make_kv_cache(B, device, hidden.dtype)
            v_cache = self._make_kv_cache(B, device, hidden.dtype)
            pos = torch.zeros(1, dtype=torch.long, device=device)
        else:
            pos = state.get("transformer_position", torch.zeros(1, dtype=torch.long, device=device))
            if k_cache[0].dtype != hidden.dtype:
                k_cache = [c.to(hidden.dtype) for c in k_cache]
                v_cache = [c.to(hidden.dtype) for c in v_cache]

        slot_t = (pos % self.horizon).long()  # (1,) long tensor

        # PE indexed by slot_t (pos resets to 0 at episode boundary via
        # pufferl.py's done handling, so PE[slot_t] = PE[pos_within_episode]).
        pos_embed = self.get_positional_embedding(self.horizon, device)  # (1, horizon, hidden)
        pos_embed_slot = pos_embed.index_select(1, slot_t).squeeze(1)  # (1, hidden)
        x = (hidden + pos_embed_slot).unsqueeze(1)  # (B, 1, hidden)

        # Build (1, 1, 1, horizon) bool mask: True at slots [0, slot_t].
        slots_arange = self._slot_arange(device)
        attn_mask = (slots_arange <= slot_t).view(1, 1, 1, self.horizon)
        H = self.num_heads
        D = self.head_dim

        for li, layer in enumerate(self.transformer.layers):
            attn = layer.self_attn

            x_norm = layer.norm1(x)
            qkv = F.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, 1, H, D).transpose(1, 2)  # (B, H, 1, D)
            k = k.view(B, 1, H, D).transpose(1, 2)  # (B, H, 1, D)
            v = v.view(B, 1, H, D).transpose(1, 2)  # (B, H, 1, D)

            # index_copy_ on dim=2 writes one slot using a tensor index, which
            # avoids the .item() sync that would force a Dynamo graph break.
            k_cache[li].index_copy_(2, slot_t, k)
            v_cache[li].index_copy_(2, slot_t, v)

            if state.get("_probe_attention", False):
                # Manual softmax attention so we can stash the weights. SDPA's
                # functional form doesn't return weights. Math is identical to
                # the SDPA call below but we capture (B, H, 1, horizon) weights
                # per layer per step into state["_attn_weights"].
                scale = 1.0 / math.sqrt(D)
                logits = torch.matmul(q, k_cache[li].transpose(-2, -1)) * scale  # (B, H, 1, horizon)
                logits = logits.masked_fill(~attn_mask, float("-inf"))
                weights = F.softmax(logits, dim=-1)  # (B, H, 1, horizon)
                attn_out = torch.matmul(weights, v_cache[li])  # (B, H, 1, D)
                state.setdefault("_attn_weights", []).append(
                    {"layer": li, "slot": int(slot_t.item()), "weights": weights.detach().cpu()}
                )
            else:
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k_cache[li],
                    v_cache[li],
                    attn_mask=attn_mask,
                    is_causal=False,
                )
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, self.hidden_size)
            attn_out = F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)
            x = x + attn_out

            x_norm2 = layer.norm2(x)
            ffn_h = layer.activation(F.linear(x_norm2, layer.linear1.weight, layer.linear1.bias))
            ffn_out = F.linear(ffn_h, layer.linear2.weight, layer.linear2.bias)
            x = x + ffn_out

        x = self.output_norm(x)
        hidden_out = x.squeeze(1)

        state["k_cache"] = k_cache
        state["v_cache"] = v_cache
        state["transformer_position"] = pos + 1
        state["hidden"] = hidden_out

        logits, values = self.policy.decode_actions(hidden_out)
        return logits, values

    def _forward_eval_legacy(self, observations, state):
        """Original full-context forward. Kept for equivalence testing only."""
        B = observations.shape[0]
        device = observations.device

        hidden = self.policy.encode_observations(observations, state=state)
        hidden = self.input_projection(hidden)

        if "transformer_context" not in state or state["transformer_context"] is None:
            context = torch.zeros(B, self.horizon, self.hidden_size, device=device, dtype=hidden.dtype)
            pos = torch.zeros(1, dtype=torch.long, device=device)
        else:
            context = state["transformer_context"]
            pos = state.get("transformer_position", torch.zeros(1, dtype=torch.long, device=device))

            if context.shape[-1] != self.hidden_size or context.shape[0] != B or context.shape[1] != self.horizon:
                context = torch.zeros(B, self.horizon, self.hidden_size, device=device, dtype=hidden.dtype)
                pos = torch.zeros(1, dtype=torch.long, device=device)
            if context.dtype != hidden.dtype:
                context = context.to(hidden.dtype)

        write_idx = (pos % self.horizon).long()
        context[:, write_idx, :] = hidden.unsqueeze(1)
        pos = pos + 1

        pos_embed = self.get_positional_embedding(self.horizon, device)
        context_with_pos = context + pos_embed
        causal_mask = self.get_causal_mask(self.horizon, device)

        output = self.transformer(context_with_pos, mask=causal_mask, is_causal=True)
        output = self.output_norm(output)

        read_idx = ((pos - 1) % self.horizon).long()
        hidden_out = output[:, read_idx, :].squeeze(1)

        state["transformer_context"] = context
        state["transformer_position"] = pos
        state["hidden"] = hidden_out

        logits, values = self.policy.decode_actions(hidden_out)
        return logits, values

    def forward(self, observations, state):
        x = observations
        device = x.device

        if x.ndim == len(self.obs_shape) + 1:
            B, T = x.shape[0], 1
        elif x.ndim == len(self.obs_shape) + 2:
            B, T = x.shape[:2]
        else:
            raise ValueError(f"Invalid input tensor shape: {x.shape}")

        x_flat = x.view(B * T, *self.obs_shape)
        hidden = self.policy.encode_observations(x_flat, state)

        hidden = hidden.view(B, T, self.input_size)
        hidden = self.input_projection(hidden)

        # Remove dynamic truncation - use clamp instead of if
        T_actual = min(T, self.horizon)  # Python int, fine
        if T_actual < T:
            hidden = hidden[:, -T_actual:]
            T = T_actual

        # Per-episode-reset PE: under multi-episode rollouts, training must
        # match rollout's PE indexing. Rollout (forward_eval) resets pos to 0
        # at every episode boundary via pufferl.py's done handling, so for
        # the same logical step within an episode, PE[pos_within_episode]
        # is added. We mirror that here: compute pos_within_episode from
        # terminals (cumsum-shifted-by-1 / cummax trick) and gather PE
        # per-slot rather than indexing 0..T-1 across the segment.
        terminals_for_pe = state.get("terminals")
        if terminals_for_pe is not None:
            pos_within_ep = self.compute_pos_within_episode(terminals_for_pe)  # (B, T) long
            pos_within_ep = pos_within_ep.clamp(max=self.horizon - 1)  # safety: long-episode guard
            # gather PE per (b, t): pe shape (1, horizon, hidden) → (B, T, hidden)
            pe_full = self.get_positional_embedding(self.horizon, device)  # (1, horizon, hidden)
            pe_per_slot = pe_full[0, pos_within_ep]  # (B, T, hidden)
            hidden = hidden + pe_per_slot.to(hidden.dtype)
        else:
            hidden = hidden + self.get_positional_embedding(T, device)

        use_episode_mask = "terminals" in state and state["terminals"] is not None

        if not use_episode_mask:
            causal_mask = self.get_causal_mask(T, device)
            if self.training and self.use_checkpointing:
                hidden = checkpoint(
                    lambda h, m: self.transformer(h, mask=m, is_causal=True), hidden, causal_mask, use_reentrant=False
                )
            else:
                hidden = self.transformer(hidden, mask=causal_mask, is_causal=True)
        else:
            terminals = state["terminals"]
            if terminals.shape[1] > T:
                terminals = terminals[:, -T:]
            causal_mask = self.get_causal_mask(T, device)
            episode_mask = self.create_episode_mask(terminals, T)
            attn_mask = causal_mask.unsqueeze(0) + episode_mask
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            if self.training and self.use_checkpointing:
                hidden = checkpoint(
                    lambda h, m: self.transformer(h, mask=m, is_causal=False), hidden, attn_mask, use_reentrant=False
                )
            else:
                hidden = self.transformer(hidden, mask=attn_mask, is_causal=False)

        hidden = self.output_norm(hidden)
        flat_hidden = hidden.contiguous().view(B * T, self.hidden_size)

        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.view(B, T)

        # Use Python int for context_len - no sync
        context_len = min(T, self.horizon)
        state["hidden"] = hidden
        state["transformer_context"] = hidden[:, -context_len:].detach()
        state["transformer_position"] = torch.full((B,), context_len - 1, dtype=torch.long, device=device)

        return logits, values


class Convolutional(nn.Module):
    def __init__(
        self,
        env,
        *args,
        framestack,
        flat_size,
        input_size=512,
        hidden_size=512,
        output_size=512,
        channels_last=False,
        downsample=1,
        **kwargs,
    ):
        """The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer

        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without."""
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        # TODO: Remove these from required params
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, :: self.downsample, :: self.downsample]
        return self.network(observations.float() / 255.0)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value


class ProcgenResnet(nn.Module):
    """Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper."""

    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2 * cnn_width, 2 * cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden

    def decode_actions(self, hidden):
        """linear decoder function"""
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
