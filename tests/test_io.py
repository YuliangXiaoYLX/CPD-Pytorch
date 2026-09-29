"""Input handling: array types, dtype and device selection, validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpd_pytorch import (
    AffineRegistration,
    DeformableRegistration,
    RigidRegistration,
    gaussian_kernel,
)
from helpers import DEVICES


def test_numpy_in_numpy_out(fish):
    X, Y = fish
    reg = RigidRegistration(X, Y)
    TY, (s, R, t) = reg.register()
    for value in (TY, s, R, t):
        assert isinstance(value, np.ndarray)
    assert TY.dtype == np.float64
    # Attributes are tensors.
    assert isinstance(reg.TY, torch.Tensor) and reg.TY.dtype == torch.float64


def test_lists_are_accepted(fish):
    X, Y = fish
    TY, _ = AffineRegistration(X.tolist(), Y.tolist()).register()
    assert isinstance(TY, np.ndarray) and TY.dtype == np.float64


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_in_tensor_out_on_input_device(device, fish):
    X, Y = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in fish)
    reg = DeformableRegistration(X, Y, max_iterations=3)
    TY, (G, W) = reg.register()
    for value in (TY, G, W):
        assert isinstance(value, torch.Tensor)
        assert value.device.type == device and value.dtype == torch.float32
    assert reg.device.type == device


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ((np.float64, np.float64), torch.float64),
        ((np.float32, np.float32), torch.float32),
        ((np.float32, np.float64), torch.float64),
        ((np.int64, np.int64), torch.float64),
        ((np.float16, np.float16), torch.float32),
    ],
)
def test_dtype_inference(inputs, expected, fish):
    X, Y = fish
    reg = RigidRegistration(X.astype(inputs[0]), Y.astype(inputs[1]))
    assert reg.dtype == expected
    assert reg.X.dtype == expected


@pytest.mark.parametrize("dtype", [torch.float32, "float32", np.float32, np.dtype("float32")])
def test_explicit_dtype_spellings(dtype, fish):
    reg = RigidRegistration(*fish, dtype=dtype, max_iterations=2)
    TY, _ = reg.register()
    assert reg.dtype == torch.float32 and TY.dtype == np.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32, "half", "complex64", object])
def test_unsupported_dtype(dtype, fish):
    with pytest.raises(TypeError, match=r"float32 or torch\.float64"):
        RigidRegistration(*fish, dtype=dtype)


def test_mps_rejects_float64(fish):
    try:
        torch.device("mps")
    except RuntimeError:
        pytest.skip("this PyTorch version has no MPS device type")
    with pytest.raises(TypeError, match="MPS"):
        RigidRegistration(*fish, device="mps", dtype=torch.float64)


def test_explicit_device_moves_tensors(fish):
    X, Y = (torch.as_tensor(a) for a in fish)
    reg = RigidRegistration(X, Y, device=torch.device("cpu"))
    assert reg.device == torch.device("cpu")


def test_inputs_are_not_modified(fish):
    X, Y = (torch.as_tensor(a) for a in fish)
    X0, Y0 = X.clone(), Y.clone()
    DeformableRegistration(X, Y, normalize=True).register()
    assert torch.equal(X, X0) and torch.equal(Y, Y0)


def test_requires_grad_inputs_are_detached(fish):
    X = torch.as_tensor(fish[0]).requires_grad_()
    TY, _ = RigidRegistration(X, torch.as_tensor(fish[1])).register()
    assert not TY.requires_grad


@pytest.mark.parametrize(
    ("X", "Y", "match"),
    [
        (np.zeros(5), np.zeros((5, 2)), "shape"),
        (np.zeros((5, 2)), np.zeros((5, 3)), "same dimension"),
        (np.zeros((0, 2)), np.zeros((5, 2)), "shape"),
        (np.full((5, 2), np.nan), np.zeros((5, 2)), "NaN"),
        (np.zeros((2, 5, 2)), np.zeros((3, 5, 2)), "Batch"),
    ],
)
def test_invalid_point_sets(X, Y, match):
    with pytest.raises(ValueError, match=match):
        RigidRegistration(X, Y)


def test_non_numeric_input():
    with pytest.raises(TypeError, match="numeric"):
        RigidRegistration([["a", "b"]], np.zeros((1, 2)))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"w": 1.0}, ValueError),
        ({"w": -0.1}, ValueError),
        ({"tolerance": -1}, ValueError),
        ({"max_iterations": -1}, ValueError),
        ({"max_iterations": "10"}, TypeError),
        ({"sigma2": -1.0}, ValueError),
        ({"chunk_size": 0}, ValueError),
    ],
)
def test_invalid_options(fish, kwargs, error):
    with pytest.raises(error):
        RigidRegistration(*fish, **kwargs)


def test_float_max_iterations_warns(fish):
    with pytest.warns(UserWarning, match="non-integer"):
        reg = RigidRegistration(*fish, max_iterations=5.5)
    assert reg.max_iterations == 5


def test_callback_receives_inputs_type(fish):
    calls = []

    def callback(iteration, error, X, Y):
        calls.append((iteration, error, type(X), type(Y)))

    reg = RigidRegistration(*fish, max_iterations=4, tolerance=0.0)
    reg.register(callback)
    assert [c[0] for c in calls] == [1, 2, 3, 4]
    assert all(isinstance(c[1], float) and c[2] is np.ndarray and c[3] is np.ndarray for c in calls)


def test_gaussian_kernel_mirrors_input_type():
    K = gaussian_kernel(np.zeros((2, 3)), 1.0)
    assert isinstance(K, np.ndarray)
    np.testing.assert_array_equal(K, np.ones((2, 2)))
    K = gaussian_kernel(torch.zeros(2, 3), 1.0, torch.ones(4, 3))
    assert isinstance(K, torch.Tensor) and K.shape == (2, 4)
    assert torch.allclose(K, torch.full((2, 4), float(np.exp(-1.5))))


def test_repr(fish):
    text = repr(RigidRegistration(*fish))
    assert text.startswith("RigidRegistration(N=91, M=91, D=2")
