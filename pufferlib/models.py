import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces

import torch.nn.functional as F
import math


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


class TransformerWrapper(nn.Module): #TransformerWrapper
    def __init__(
        self,
        env,
        policy,
        input_size=128,
        hidden_size=128,
        num_layers=4,
        num_heads=8,
        context_length=512,
        dropout=0.0,
    ):
        """Wraps your policy with a Transformer for temporal modeling.
        
        Args:
            env: Environment instance
            policy: Your Drive policy (must have encode_observations and decode_actions)
            input_size: Size of encoded observations (from policy.encode_observations)
            hidden_size: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            context_length: Maximum sequence length to attend over
            dropout: Dropout probability
        """
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.is_continuous = self.policy.is_continuous

        # Project encoded observations to transformer dimension if needed
        if input_size != hidden_size:
            self.input_projection = nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = nn.Identity()

        # Learnable positional embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, context_length, hidden_size)
        )
        nn.init.normal_(self.positional_embedding, std=0.02)

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
            mask = self.create_causal_mask(T, 'cpu')
            self.register_buffer(f'_causal_mask_{T}', mask, persistent=False)

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
        # Upper triangular matrix of -inf (mask future positions)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask

    def get_causal_mask(self, T, device):
        """Get cached causal mask or create new one"""
        buffer_name = f'_causal_mask_{T}'
        if hasattr(self, buffer_name):
            return getattr(self, buffer_name).to(device)
        # Fallback for uncommon lengths
        return self.create_causal_mask(T, device)

    def create_episode_mask(self, terminals, seq_len):
        """Create mask that prevents attending across episode boundaries
        
        Args:
            terminals: [B, T] tensor of terminal flags
            seq_len: sequence length T
            
        Returns:
            mask: [B, T, T] attention mask
        """
        B = terminals.shape[0]
        device = terminals.device
        
        # Compute episode IDs by cumulative sum of terminals
        # Shift by 1 so terminal states are in the same episode as preceding states
        episode_ids = torch.cat([
            torch.zeros(B, 1, device=device),
            terminals[:, :-1]
        ], dim=1).cumsum(dim=1)
        
        # Create mask: can only attend within same episode
        # [B, T, 1] == [B, 1, T] -> [B, T, T]
        episode_mask = episode_ids.unsqueeze(2) == episode_ids.unsqueeze(1)
        
        # Convert to additive mask (-inf where attention not allowed)
        mask = torch.zeros(B, seq_len, seq_len, device=device)
        mask.masked_fill_(~episode_mask, float('-inf'))
        
        return mask

    def forward_eval(self, observations, state):
        """Optimized forward for eval - minimal context"""
        B = observations.shape[0]
        device = observations.device
        
        # Encode observations
        hidden = self.policy.encode_observations(observations, state=state)
        hidden = self.input_projection(hidden)
        
        # For eval, just use last N observations (don't accumulate unbounded)
        if "transformer_context" not in state or state["transformer_context"] is None:
            # Start with just current observation
            context = hidden.unsqueeze(1)  # [B, 1, D]
        else:
            prev_context = state["transformer_context"]
            
            # CHECK: If cached context has wrong shape, reinitialize
            if prev_context.shape[-1] != self.hidden_size:
                print(f"Warning: Cached context has wrong shape {prev_context.shape[-1]}, expected {self.hidden_size}. Reinitializing.")
                context = hidden.unsqueeze(1)
            else:
                context = torch.cat([prev_context, hidden.unsqueeze(1)], dim=1)
                # Keep only last context_length observations
                if context.shape[1] > self.context_length:
                    context = context[:, -self.context_length:, :]
        
        seq_len = context.shape[1]
        
        # Skip transformer for very short sequences
        if seq_len <= 2:
            # Not enough context for transformer to help
            hidden = context[:, -1, :]
        else:
            # Add positional embeddings
            context = context + self.positional_embedding[:, :seq_len, :]
            
            # Simple causal mask
            causal_mask = self.create_causal_mask(seq_len, device)
            
            # Apply transformer
            output = self.transformer(context, mask=causal_mask, is_causal=True)
            output = self.output_norm(output)
            hidden = output[:, -1, :]
        
        # Update state
        state["transformer_context"] = context.detach()
        state["hidden"] = hidden
        
        # Decode
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        """Forward function for training - optimized version
        
        Processes sequences in parallel for efficient training.
        """
        x = observations
        device = x.device
        
        # Simplified shape parsing
        if x.ndim == len(self.obs_shape) + 1:
            B, T = x.shape[0], 1
        elif x.ndim == len(self.obs_shape) + 2:
            B, T = x.shape[:2]
        else:
            raise ValueError(f"Invalid input tensor shape: {x.shape}")
        
        # Flatten batch and time for encoding
        x_flat = x.view(B * T, *self.obs_shape)
        hidden = self.policy.encode_observations(x_flat, state)
        
        # Reshape to [B, T, D] and project
        hidden = hidden.view(B, T, self.input_size)
        hidden = self.input_projection(hidden)
        
        # Truncate to context length if needed
        if T > self.context_length:
            hidden = hidden[:, -self.context_length:]
            T = self.context_length
        
        # Add positional embeddings (cached slice)
        hidden = hidden + self.positional_embedding[:, :T]
        
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Handle episode boundary mask if needed
        if "terminals" in state and state["terminals"] is not None:
            terminals = state["terminals"]
            if terminals.shape[1] > T:
                terminals = terminals[:, -T:]
            
            episode_mask = self.create_episode_mask(terminals, T)
            
            # Combine masks
            attn_mask = causal_mask.unsqueeze(0) + episode_mask  # (B, T, T)
            
            # Optimized expansion for multi-head attention
            # More efficient than reshape -> expand -> reshape
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            is_causal = False
        else:
            attn_mask = causal_mask
            is_causal = True
        
        # Apply transformer
        hidden = self.transformer(hidden, mask=attn_mask, is_causal=is_causal)
        hidden = self.output_norm(hidden)
        
        # Flatten for action decoding (use view instead of reshape when possible)
        flat_hidden = hidden.contiguous().view(B * T, self.hidden_size)
        
        # Decode actions
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.view(B, T)
        
        # Update state (compute context length once)
        context_len = min(T, self.context_length)
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
