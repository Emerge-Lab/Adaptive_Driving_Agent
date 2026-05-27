"""Decisive test: does batch size affect row 0's output in the streaming
attention pattern under fp32 on CUDA?

We replicate the EXACT operation streaming forward_eval does:
- Linear projection (Q, K, V from x_norm)
- F.scaled_dot_product_attention with cached K/V

For the SAME row-0 input, run at B=1, B=64, B=512, B=2048, and compare row 0's
output bit-for-bit.

If outputs match exactly → batch size does NOT change row 0's math.
If outputs differ → batch size DOES change row 0's math (kernel selection).

This is THE definitive test of my "batch-size-dependent kernel" hypothesis.
"""
import os
import torch
import torch.nn.functional as F


def main():
    device = "cuda"
    dtype = torch.float32

    # Match the actual policy architecture
    H = 4         # num_heads
    D = 64        # head_dim
    hidden_size = H * D  # 256
    T_max = 402   # full horizon (cache length)
    T_steps = 50  # number of timesteps to simulate

    torch.manual_seed(42)

    # Deterministic projection weights (one layer's QKV linear)
    in_proj_weight = torch.randn(3 * hidden_size, hidden_size, device=device, dtype=dtype)
    in_proj_bias = torch.randn(3 * hidden_size, device=device, dtype=dtype)

    # Deterministic obs hidden for row 0 at each step
    torch.manual_seed(0)
    x_row0 = torch.randn(T_steps, hidden_size, device=device, dtype=dtype)

    # Try matmul precision settings
    torch.set_float32_matmul_precision("highest")  # disables TF32 in matmul

    def run_streaming(B):
        """Replicate the streaming forward_eval attention pattern at batch B.
        Row 0 input is identical across all B values. Other rows are random
        (but seeded the same way each call).
        """
        torch.manual_seed(123)
        other_x = torch.randn(T_steps, B - 1, hidden_size, device=device, dtype=dtype) if B > 1 else None

        # Allocate K/V cache
        k_cache = torch.zeros(B, H, T_max, D, device=device, dtype=dtype)
        v_cache = torch.zeros(B, H, T_max, D, device=device, dtype=dtype)

        row0_outputs = []
        for t in range(T_steps):
            if B > 1:
                x_t = torch.cat([x_row0[t : t + 1], other_x[t]], dim=0)  # (B, hidden)
            else:
                x_t = x_row0[t : t + 1]  # (1, hidden)

            # Layer norm before projection (skip for this isolated test — we
            # care about projection + attention only)
            qkv = F.linear(x_t, in_proj_weight, in_proj_bias)  # (B, 3*hidden)
            q, k, v = qkv.chunk(3, dim=-1)  # each (B, hidden)
            q = q.view(B, 1, H, D).transpose(1, 2)  # (B, H, 1, D)
            k = k.view(B, 1, H, D).transpose(1, 2)
            v = v.view(B, 1, H, D).transpose(1, 2)

            # Write to cache at slot t
            slot_t = torch.tensor([t], device=device, dtype=torch.long)
            k_cache.index_copy_(2, slot_t, k)
            v_cache.index_copy_(2, slot_t, v)

            # Build the same boolean mask the streaming path uses
            slots_arange = torch.arange(T_max, device=device)
            base_mask = (slots_arange <= slot_t).view(1, T_max)
            garbage_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
            attn_mask = (base_mask & ~garbage_mask).view(B, 1, 1, T_max)

            attn_out = F.scaled_dot_product_attention(
                q, k_cache, v_cache,
                attn_mask=attn_mask,
                is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, hidden_size)
            row0_outputs.append(attn_out[0, 0].detach().cpu())  # (hidden,)

        return torch.stack(row0_outputs)  # (T_steps, hidden)

    print(f"matmul precision: {torch.get_float32_matmul_precision()}")
    print(f"NVIDIA_TF32_OVERRIDE: {os.environ.get('NVIDIA_TF32_OVERRIDE', '(unset)')}")
    print(f"PUFFER_TRANSFORMER_LEGACY_EVAL: {os.environ.get('PUFFER_TRANSFORMER_LEGACY_EVAL', '(unset)')}")
    print()
    out_B1 = run_streaming(1)
    print(f"{'B':>6}  {'identical?':>12}  {'max diff':>12}  {'mean diff':>12}  {'first_t_diverge':>16}")
    for B in [1, 4, 16, 64, 256, 512, 2048]:
        out = run_streaming(B)
        diff = (out_B1 - out).abs()
        per_t = diff.max(dim=1).values
        first = -1
        for t in range(T_steps):
            if per_t[t].item() > 0:
                first = t
                break
        ident = (out_B1 == out).all().item()
        print(f"{B:>6}  {str(ident):>12}  {diff.max().item():>12.8f}  {diff.mean().item():>12.8f}  {first:>16}")


if __name__ == "__main__":
    main()
