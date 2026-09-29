"""Shared test utilities."""

from __future__ import annotations

import os

import numpy as np
import torch


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


#: Devices available on this machine; GPU tests run wherever a GPU exists.
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if _mps_available():
    DEVICES.append("mps")
# CPD_PYTORCH_TEST_DEVICES="cpu,cuda" restricts the list, e.g. on CI machines whose GPU
# is reported as available but cannot run PyTorch's kernels.
_requested = os.environ.get("CPD_PYTORCH_TEST_DEVICES", "").strip()
if _requested:
    DEVICES = [d for d in DEVICES if d in {r.strip() for r in _requested.split(",")}]

#: (device, dtype) combinations supported here (MPS has no float64).
DEVICE_DTYPES: list[tuple[str, torch.dtype]] = [
    (device, dtype)
    for device in DEVICES
    for dtype in (torch.float32, torch.float64)
    if not (device == "mps" and dtype == torch.float64)
]


def lift_to_3d(points: np.ndarray) -> np.ndarray:
    """Stack a 2D shape at z=0 and z=1, as in the original 3D examples."""
    n = len(points)
    return np.vstack((np.c_[points, np.zeros(n)], np.c_[points, np.ones(n)]))


def rotation(*angles: float) -> np.ndarray:
    """2D rotation for one angle; 3D rotation Rz(a) Ry(b) Rx(c) for three angles."""
    if len(angles) == 1:
        (a,) = angles
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    a, b, c = angles
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def numpy(value: object) -> np.ndarray:
    """Convert a tensor or array to a float64 NumPy array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def max_rel_error(actual: object, expected: object) -> float:
    a, b = numpy(actual), numpy(expected)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
