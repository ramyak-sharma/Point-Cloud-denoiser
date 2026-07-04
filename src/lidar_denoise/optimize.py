"""Step 5: real-time optimisation — profiling, dynamic quantisation, batching.

The proposal targets the 10 Hz scan rate of automotive LiDAR. This module
measures per-frame latency for every method and applies PyTorch dynamic
quantisation (float32 -> int8 weights on Linear/Conv layers, CPU execution)
to the learning models. Throughput numbers are reported as measured — no
extrapolation.
"""

from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import torch


def profile_callable(fn, inputs: list[np.ndarray], warmup: int = 1) -> dict:
    """Run fn over each input cloud; report per-frame latency stats and Hz."""
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "hz": float(1000.0 / times.mean()),
        "n_frames": len(times),
    }


def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """int8 dynamic quantisation of Linear layers (CPU backend).

    Conv1d/Conv2d dynamic quantisation is not supported by PyTorch's dynamic
    backend, so EdgeConv convolutions stay float32; the FC-heavy PointNet AE
    decoder benefits the most.
    """
    # macOS arm64 torch ships only the qnnpack engine, and none is selected
    # by default — without this, linear_prepack raises "NoQEngine".
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    cpu_model = deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(
        cpu_model, {torch.nn.Linear}, dtype=torch.qint8
    )


def model_size_mb(model: torch.nn.Module) -> float:
    """Serialised parameter size in MB (state_dict on disk)."""
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6
