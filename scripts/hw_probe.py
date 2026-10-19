"""Run first on a new model box: capacity, integrity, speed per dtype, link bandwidth.

    PYTHONPATH=. python scripts/hw_probe.py [fill_fraction]   # default 0.9 of free VRAM

Also prints `nvidia-smi` PCIe link generation/width if available. Paste the
output back into the session; it decides dtype and model placement.
"""
import json
import shutil
import subprocess
import sys

import torch

from blammoe.hw import copy_gbps, device_summary, matmul_tflops, memory_integrity


def main(fill_fraction: float = 0.9) -> None:
    report = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "cuda_version": torch.version.cuda, "devices": []}
    if shutil.which("nvidia-smi"):
        q = "name,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv"], capture_output=True, text=True)
        report["nvidia_smi"] = r.stdout.strip().splitlines()
    cpu = torch.device("cpu")
    report["cpu"] = {"fp32_tflops": round(matmul_tflops(cpu, torch.float32, n=2048, iters=3), 3)}
    devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    for d in devs:
        info = device_summary(d)
        print(f"probing {info}", flush=True)
        try:
            info["arch_supported_by_this_torch"] = f"sm_{info['capability'].replace('.', '')}" in torch.cuda.get_arch_list()
            info["tflops"] = {str(dt).split(".")[-1]: round(matmul_tflops(d, dt), 2)
                              for dt in (torch.float32, torch.float16, torch.bfloat16)}
        except RuntimeError as e:
            info["tflops_error"] = str(e).splitlines()[0]
        info["h2d_gbps"] = round(copy_gbps(cpu, d), 2)
        info["d2h_gbps"] = round(copy_gbps(d, cpu), 2)
        info["memory_integrity"] = memory_integrity(d, gb=info["free_gb"] * fill_fraction)
        report["devices"].append(info)
        print(json.dumps(info, indent=1), flush=True)
    if len(devs) >= 2:
        report["d2d_gbps"] = {f"{a}->{b}": round(copy_gbps(a, b), 2) for a in devs for b in devs if a != b}
        report["p2p_access"] = {f"{a}->{b}": torch.cuda.can_device_access_peer(a.index, b.index)
                                for a in devs for b in devs if a != b}
    print("\n=== REPORT ===\n" + json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:]))
