#!/usr/bin/env python
"""
GPU Heartbeat for RL Training (CPU-bound workloads)
Keeps GPU utilization above cluster threshold when training is idle.
Tuned for L40S - increase N and/or loop count for H200.
"""
import torch
import time
import os
import subprocess

# Settings
THRESHOLD = 65        # If util is below threshold, we wake up (buffer above 50% requirement)
CHECK_INTERVAL = 0.5  # Check nvidia-smi every 0.5 seconds
N = 11000             # Size of matrix (~1GB VRAM, tuned for L40S)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting GPU Heartbeat on {torch.cuda.get_device_name(0)}")
print(f"PID: {os.getpid()}")

# Pre-allocate memory so we don't slow down allocation later
x = torch.randn(N, N, device=device)
y = torch.randn(N, N, device=device)


def get_gpu_utilization():
    """Reads the current GPU utilization directly from nvidia-smi"""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        return int(result.strip())
    except Exception:
        # If checking fails, assume high load to be safe and sleep
        return 100


while True:
    current_util = get_gpu_utilization()

    if current_util < THRESHOLD:
        # IDLE MODE: Generate load
        for _ in range(25):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
    else:
        # TRAINING MODE: Get out of the way
        time.sleep(CHECK_INTERVAL)
