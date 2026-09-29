"""Benchmark cpd-pytorch against pycpd and, optionally, the legacy ``torchcpd`` code.

Every case runs a fixed number of EM iterations in a fresh subprocess, so peak memory is
measured per case. Each case first runs an untimed warm-up (the same method on up to 2,000
points, 2 iterations), so one-time costs such as CUDA context creation, library handles and
kernel loading stay out of the timings. The timed run includes the per-registration setup
(kernel matrix, low-rank eigenpairs). Results are printed as a Markdown table.

Examples
--------
Compare with pycpd on the CPU in float64::

    python benchmarks/benchmark.py

GPU, float32, larger problems::

    python benchmarks/benchmark.py --device cuda --dtype float32 --sizes 2000 8000 20000

To pick one GPU on a multi-GPU machine, prefer ``CUDA_VISIBLE_DEVICES`` over ``cuda:N``::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python benchmarks/benchmark.py --device cuda

Also time the pre-1.0 ``torchcpd`` package from an old checkout::

    python benchmarks/benchmark.py --legacy /path/to/old/CPD-Pytorch
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time

DEFAULT_SIZES = {
    "rigid": [1000, 4000, 8000],
    "affine": [1000, 4000, 8000],
    "deformable": [500, 1000, 2000],
    "deformable-low-rank": [2000, 5000],
}
CLASSES = {
    "rigid": "RigidRegistration",
    "affine": "AffineRegistration",
    "deformable": "DeformableRegistration",
    "deformable-low-rank": "DeformableRegistration",
}


def _synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _peak_memory_mb(device: str) -> float:
    import torch

    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated(torch.device(device)) / 2**20
    if device.startswith("mps"):  # no peak-memory counter for MPS
        return float("nan")
    try:
        import resource
    except ImportError:  # Windows
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes vs KiB


def _register(args: argparse.Namespace, size: int, iterations: int) -> None:
    """Register a random 3D point set of ``size`` points with its rotated, noisy copy."""
    import numpy as np
    import torch

    rng = np.random.default_rng(0)
    X = rng.normal(size=(size, 3))
    angle = 0.2
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    Y = X @ R + 0.1 + rng.normal(0, 0.01, X.shape)
    kwargs = {"max_iterations": iterations, "tolerance": 0.0}
    if args.method == "deformable-low-rank":
        kwargs.update(low_rank=True, num_eig=100)
    dtype = getattr(torch, args.dtype)

    if args.impl == "cpd-pytorch":
        import cpd_pytorch as module

        X_in = torch.as_tensor(X, dtype=dtype, device=args.device)
        Y_in = torch.as_tensor(Y, dtype=dtype, device=args.device)
    elif args.impl == "legacy":
        sys.path.insert(0, args.legacy)
        import torchcpd as module

        X_in, Y_in = X.astype(np.float64), Y.astype(np.float64)
        kwargs.update(device=args.device, dtype=dtype)
    else:
        import pycpd as module

        X_in, Y_in = X, Y

    getattr(module, CLASSES[args.method])(X=X_in, Y=Y_in, **kwargs).register()
    _synchronize(args.device)


def worker(args: argparse.Namespace) -> None:
    """Run one case and print a JSON line with the timing and memory."""
    import torch

    _register(args, min(args.size, 2000), iterations=2)  # untimed warm-up
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    start = time.perf_counter()
    _register(args, args.size, args.iterations)
    elapsed = time.perf_counter() - start
    print(json.dumps({"seconds": elapsed, "peak_mb": _peak_memory_mb(args.device)}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--methods", nargs="+", default=list(DEFAULT_SIZES), choices=list(DEFAULT_SIZES)
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, help="point counts N = M (default: per method)"
    )
    parser.add_argument("--legacy", help="path to a checkout of the pre-1.0 torchcpd code")
    parser.add_argument("--no-pycpd", action="store_true", help="skip pycpd (CPU, float64 only)")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per case")
    # Worker mode (internal).
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--impl", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return

    impls = ["cpd-pytorch"]
    if args.legacy:
        impls.append("legacy")
    if not args.no_pycpd and args.device == "cpu" and args.dtype == "float64":
        impls.append("pycpd")

    name = args.device
    if args.device.startswith("cuda"):
        import torch

        name = f"{args.device} ({torch.cuda.get_device_name(torch.device(args.device))})"
    print(f"device={name} dtype={args.dtype} iterations={args.iterations}\n")
    print("| method | N = M | implementation | time / iteration | peak memory |")
    print("|---|---:|---|---:|---:|")
    for method in args.methods:
        for size in args.sizes or DEFAULT_SIZES[method]:
            for impl in impls:
                command = [
                    sys.executable, __file__, "--worker", "--impl", impl, "--method", method,
                    "--size", str(size), "--device", args.device, "--dtype", args.dtype,
                    "--iterations", str(args.iterations),
                ]  # fmt: skip
                if args.legacy:
                    command += ["--legacy", args.legacy]
                try:
                    run = subprocess.run(
                        command, capture_output=True, text=True, timeout=args.timeout, check=True
                    )
                    result = json.loads(run.stdout.strip().splitlines()[-1])
                    per_iteration = f"{1000 * result['seconds'] / args.iterations:,.1f} ms"
                    peak = result["peak_mb"]
                    memory = "n/a" if math.isnan(peak) else f"{peak:,.0f} MB"
                except subprocess.TimeoutExpired:
                    per_iteration, memory = "timeout", "-"
                except subprocess.CalledProcessError as error:
                    last = (error.stderr.strip().splitlines() or ["failed"])[-1]
                    per_iteration, memory = f"error: {last[:60]}", "-"
                print(f"| {method} | {size:,} | {impl} | {per_iteration} | {memory} |", flush=True)


if __name__ == "__main__":
    main()
