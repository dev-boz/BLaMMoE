"""Hardware probe for the model boxes (WP1 prerequisite).

Unlocked / modded cards (e.g. memory-unlocked CMP 170HX) can report capacity or
link speed they do not reliably deliver, and may have throttled math in some
precisions. Before a multi-hour run, check:

* ``memory_integrity`` — fill most of the reported memory with a known pattern
  and read it back. Catches capacity that is reported but not real or stable.
* ``matmul_tflops``    — achieved matmul throughput per dtype, to pick the fast one.
* ``copy_gbps``        — host<->device and device<->device bandwidth (real link speed).

Everything is device-agnostic and runs on CPU too (that is how it is tested).
"""
from __future__ import annotations

import time

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_summary(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"device": str(device), "type": "cpu", "threads": torch.get_num_threads()}
    p = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"device": str(device), "name": p.name, "capability": f"{p.major}.{p.minor}",
            "total_gb": round(total / 1e9, 2), "free_gb": round(free / 1e9, 2),
            "sms": p.multi_processor_count}


def memory_integrity(device: torch.device, gb: float, chunk_gb: float = 1.0) -> dict:
    """Allocate ``gb`` in chunks, write a per-chunk pattern, read every chunk back."""
    n = int(chunk_gb * 1e9 // 4)
    chunks, t0 = [], time.time()
    try:
        for i in range(max(1, int(gb / chunk_gb))):
            c = torch.empty(n, dtype=torch.int32, device=device)
            c.copy_(torch.arange(n, dtype=torch.int32, device=device) ^ (i * 2654435761 & 0x7FFFFFFF))
            chunks.append(c)
        _sync(device)
        bad = 0
        for i, c in enumerate(chunks):
            expect = torch.arange(n, dtype=torch.int32, device=device) ^ (i * 2654435761 & 0x7FFFFFFF)
            bad += int((c != expect).sum())
        return {"requested_gb": gb, "allocated_gb": round(len(chunks) * n * 4 / 1e9, 2),
                "mismatches": bad, "ok": bad == 0, "sec": round(time.time() - t0, 1)}
    except RuntimeError as e:  # OOM or driver error: report how far it got
        return {"requested_gb": gb, "allocated_gb": round(len(chunks) * n * 4 / 1e9, 2),
                "ok": False, "error": str(e).splitlines()[0]}
    finally:
        del chunks
        if device.type == "cuda":
            torch.cuda.empty_cache()


def matmul_tflops(device: torch.device, dtype: torch.dtype, n: int = 4096, iters: int = 10) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    a @ b  # warm-up
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    _sync(device)
    return 2 * n**3 * iters / (time.perf_counter() - t0) / 1e12


def copy_gbps(src: torch.device, dst: torch.device, mb: int = 512, iters: int = 5) -> float:
    x = torch.empty(mb * 2**20, dtype=torch.uint8, device=src)
    if src.type == "cpu" and dst.type == "cuda":
        x = x.pin_memory()
    move = (lambda t: t.clone()) if src == dst else (lambda t: t.to(dst))  # .to(same) is a no-op
    move(x)  # warm-up
    _sync(dst)
    t0 = time.perf_counter()
    for _ in range(iters):
        move(x)
    _sync(dst)
    _sync(src)
    return mb * 2**20 * iters / (time.perf_counter() - t0) / 1e9
