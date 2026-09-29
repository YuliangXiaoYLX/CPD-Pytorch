"""Conversion between user inputs (NumPy / PyTorch) and the tensors used internally."""

from __future__ import annotations

import functools
from typing import Any, Union

import numpy as np
import torch

__all__ = [
    "ArrayLike",
    "DTypeLike",
    "DeviceLike",
    "as_output",
    "as_tensor",
    "check_dtype_device",
    "is_numpy_like",
    "resolve_device",
    "resolve_dtype",
    "to_tensor",
]

#: Anything convertible to a numeric array: a ``numpy.ndarray``, a ``torch.Tensor``,
#: or a (nested) sequence of numbers.
ArrayLike = Union[np.ndarray, torch.Tensor, Any]
#: A floating dtype: ``torch.float32``/``torch.float64``, ``np.float32``/``np.float64``,
#: or the strings ``"float32"``/``"float64"``.
DTypeLike = Union[torch.dtype, np.dtype, type, str]
#: A device: ``torch.device`` or a string such as ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``.
DeviceLike = Union[torch.device, str]

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_NUMPY_TO_TORCH = {np.dtype("float32"): torch.float32, np.dtype("float64"): torch.float64}


def resolve_dtype(dtype: DTypeLike | None, *inputs: torch.Tensor) -> torch.dtype:
    """Return the floating dtype used for the computation.

    An explicit ``dtype`` must be float32 or float64. Otherwise the dtype is inferred
    from the (already converted) inputs: float32 and float64 are kept (mixed inputs
    promote to float64), half precision is upcast to float32, and integer inputs
    use float64.
    """
    if dtype is not None:
        resolved: torch.dtype | None
        if isinstance(dtype, torch.dtype):
            resolved = dtype
        else:
            try:
                resolved = _NUMPY_TO_TORCH.get(np.dtype(dtype))
            except TypeError:
                resolved = None
        if resolved not in _SUPPORTED_DTYPES:
            raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype!r}.")
        assert resolved is not None
        return resolved

    floating = [t.dtype for t in inputs if t.is_floating_point()]
    if not floating:
        return torch.float64
    promoted = functools.reduce(torch.promote_types, floating)
    return promoted if promoted in _SUPPORTED_DTYPES else torch.float32


def resolve_device(device: DeviceLike | None, *inputs: Any) -> torch.device:
    """Return ``device`` if given, else the device of the first tensor input, else the CPU."""
    if device is not None:
        return torch.device(device)
    for value in inputs:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def check_dtype_device(dtype: torch.dtype, device: torch.device) -> None:
    """Reject combinations the backend cannot run."""
    if device.type == "mps" and dtype == torch.float64:
        raise TypeError(
            "The MPS backend does not support float64; pass dtype=torch.float32 "
            "(or use device='cpu' for float64)."
        )


def as_tensor(value: ArrayLike, name: str) -> torch.Tensor:
    """Convert ``value`` to a tensor without changing its dtype or device.

    Tensors are detached (registration is not differentiable). NumPy arrays and
    sequences go through ``numpy.asarray``, so Python lists of floats become float64
    as they would in NumPy.
    """
    if isinstance(value, torch.Tensor):
        return value.detach()
    try:
        array = np.asarray(value)
        return torch.as_tensor(array)
    except (TypeError, ValueError, RuntimeError) as err:
        raise TypeError(
            f"{name} must be a numeric array or tensor, got {type(value).__name__}."
        ) from err


def to_tensor(
    value: ArrayLike, name: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Convert ``value`` to a tensor with the given dtype and device."""
    return as_tensor(value, name).to(device=device, dtype=dtype)


def is_numpy_like(value: Any) -> bool:
    """Whether ``value`` should be answered with NumPy arrays (anything that is not a tensor)."""
    return not isinstance(value, torch.Tensor)


def as_output(value: Any, numpy: bool) -> Any:
    """Convert tensors in ``value`` (possibly nested in tuples) to NumPy if ``numpy`` is true."""
    if isinstance(value, tuple):
        return tuple(as_output(v, numpy) for v in value)
    if numpy and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value
