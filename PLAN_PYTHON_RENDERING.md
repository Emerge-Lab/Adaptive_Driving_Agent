# Plan: Port Python-Based Rendering from PufferDrive 3.0

## Problem Statement

Currently, rendering in Adaptive_Driving_Agent only works for LSTM policies because:
1. The `visualize.c` binary performs both **policy inference** and **rendering** in C
2. The C neural network code (`drivenet.h`) is hardcoded for LSTM architecture
3. When training with Transformers, the exported weights don't match what the C code expects

## Solution from PufferDrive 3.0

PufferDrive 3.0 decouples policy inference from rendering:
1. **Policy inference** → Done in Python/PyTorch (works with ANY architecture)
2. **Environment simulation** → Done via C bindings
3. **Graphics rendering** → Done via C bindings (`vec_render`)

This means the `visualize.c` standalone binary is bypassed, and rendering happens directly during Python rollouts using PyTorch models.

## Key Changes Required

### Phase 1: C/Binding Layer Updates

#### 1.1 Update `drive.h` - Add render_mode and rendering support
- Add `render_mode` field to `Drive` struct (RENDER_OFF=0, RENDER_HEADLESS=1, RENDER_WINDOW=2)
- Add rendering state fields (VideoRecorder, Client, PBO buffers)
- Port `c_render()` function from PufferDrive 3.0 for headless rendering
- Add video file suffix support for multi-view rendering

#### 1.2 Update `env_binding.h` - Add rendering bindings
- Update `vec_render()` to accept view_mode, draw_traces, env_id parameters
- Add `vec_set_video_suffix()` binding
- Add `RenderView` enum (FULL_SIM_STATE=0, BEV=1, PERSPECTIVE=2)

#### 1.3 Update `binding.c` - Implement the new bindings
- Wire up the new rendering functions to Python

### Phase 2: Python Layer Updates

#### 2.1 Create `rollout.py` - Unified rollout loop
Port from PufferDrive 3.0:
```python
# pufferlib/ocean/drive/rollout.py
@dataclass
class RenderContext:
    view_mode: int
    env_id: int = 0
    draw_traces: bool = True
    video_suffix: str = ""

def rollout_loop(policy, env, device, use_rnn, max_steps, render_ctx):
    """Run policy rollout with optional rendering.

    Policy inference happens in Python/PyTorch - works with any model.
    Rendering happens via C bindings - env.driver_env.render()
    """
```

#### 2.2 Update `drive.py` - Add render method and RenderView enum
```python
class RenderView(IntEnum):
    FULL_SIM_STATE = 0
    BEV = 1
    PERSPECTIVE = 2

class Drive:
    def render(self, view_mode: RenderView, draw_traces: bool, env_id: int):
        binding.vec_render(self.c_envs, int(view_mode), draw_traces, env_id)

    def set_video_suffix(self, suffix: str, env_id: int = 0):
        binding.vec_set_video_suffix(self.c_envs, env_id, suffix)
```

#### 2.3 Update `pufferl.py` - Use Python rendering instead of C visualize binary
Replace the current approach that:
- Exports model to `.bin`
- Calls `./visualize` subprocess

With new approach that:
- Uses `rollout_loop()` with `RenderContext`
- Policy inference via PyTorch (supports LSTM/Transformer)
- Rendering via `driver.render()` (C binding)

#### 2.4 Update `utils.py` - Update render_videos() function
- Remove dependency on `./visualize` binary
- Use the new Python-based rendering approach
- Keep the wandb video logging logic

### Phase 3: Evaluator Updates (Optional)

#### 3.1 Create SafeEvaluator class (if needed)
Port the evaluation and rendering logic from PufferDrive 3.0's evaluator.py:
- Safe evaluation with rendering
- Video logging to wandb

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `pufferlib/ocean/drive/drive.h` | Modify | Add render_mode, c_render(), video recording |
| `pufferlib/ocean/env_binding.h` | Modify | Add vec_render(view, traces, env), vec_set_video_suffix() |
| `pufferlib/ocean/drive/drive.py` | Modify | Add render(), set_video_suffix(), RenderView enum |
| `pufferlib/ocean/drive/rollout.py` | Create | Port rollout_loop from PufferDrive 3.0 |
| `pufferlib/pufferl.py` | Modify | Use Python rendering instead of C visualize binary |
| `pufferlib/utils.py` | Modify | Update render_videos() to use new approach |

## Implementation Order

1. **C Layer** (drive.h, env_binding.h) - Foundation for rendering
2. **Python Bindings** (drive.py) - Expose render to Python
3. **Rollout Module** (rollout.py) - Policy inference + rendering loop
4. **Integration** (pufferl.py, utils.py) - Connect to training loop
5. **Testing** - Verify Transformer renders work

## Key Benefits

1. **Architecture Agnostic**: Works with LSTM, Transformer, or any future architecture
2. **Unified Code Path**: Single rendering approach for all policies
3. **Better Performance**: GPU-accelerated headless rendering (EGL)
4. **Easier Maintenance**: No need to maintain separate C neural network code

## Testing Plan

1. Train a Transformer co-player model
2. Verify rendering produces valid videos during training
3. Verify videos are logged to wandb correctly
4. Compare render quality/performance with LSTM baseline

## References

- PufferDrive 3.0 commits:
  - `4d15f29d` - "Implementing New Renders in 3.0"
  - `fe68642c` - "GPU-accelerated headless rendering (32x speedup)"
