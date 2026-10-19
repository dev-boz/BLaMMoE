"""hw probe helpers run on CPU (the same code paths the GPU boxes use)."""
import torch

from blammoe.hw import copy_gbps, device_summary, matmul_tflops, memory_integrity

CPU = torch.device("cpu")


def test_memory_integrity_reads_back_pattern():
    r = memory_integrity(CPU, gb=0.02, chunk_gb=0.01)
    assert r["ok"] and r["mismatches"] == 0 and r["allocated_gb"] > 0.015


def test_throughput_and_bandwidth_are_positive():
    assert matmul_tflops(CPU, torch.float32, n=128, iters=2) > 0
    assert copy_gbps(CPU, CPU, mb=4, iters=2) > 0
    assert device_summary(CPU)["type"] == "cpu"
