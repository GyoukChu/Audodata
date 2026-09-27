#!/usr/bin/env python3
"""Keep every GPU's memory utilization above the cluster's idle threshold (1.5% for 3 h => container deleted).

Allocates ~KEEPALIVE_GB on each visible GPU (default 3.5 GB ≈ 1.9% of a 183 GB B200) and holds it forever, touching
the buffer periodically. Small enough not to disturb the vLLM memory layout (GLM 0.80 + solver 0.16 + this < 1.0).
Run detached:  setsid nohup .venv/bin/python scripts/gpu_keepalive.py > logs/gpu_keepalive.log 2>&1 &
"""
import os
import time

import torch

gb = float(os.environ.get("KEEPALIVE_GB", "3.5"))
bufs = []
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        n = int(gb * (1024 ** 3) / 4)
        bufs.append(torch.ones(n, dtype=torch.float32, device=f"cuda:{i}"))
        print(f"gpu{i}: holding {gb} GB", flush=True)
while True:
    for i, b in enumerate(bufs):
        b[:1024].add_(1.0)  # touch
    torch.cuda.synchronize()
    time.sleep(300)
