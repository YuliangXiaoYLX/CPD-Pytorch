from __future__ import annotations

import numpy as np
import pytest
import torch

from cpd_pytorch import AffineRegistration
from helpers import DEVICE_DTYPES, lift_to_3d, numpy


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
@pytest.mark.parametrize("shape", ["fish", "bunny"])
def test_recovers_affine_transform(device, dtype, shape, rng, request):
    # A real, asymmetric shape: an isotropic random blob is ambiguous under affine maps.
    X = request.getfixturevalue(shape)[0] - 5.0
    D = X.shape[1]
    B_true = np.eye(D) + 0.2 * rng.normal(size=(D, D))
    t_true = rng.normal(size=D)
    Y = (X - t_true) @ np.linalg.inv(B_true)  # so that X = Y @ B_true + t_true
    reg = AffineRegistration(X, Y, device=device, dtype=dtype)
    TY, (B, t) = reg.register()
    assert isinstance(TY, np.ndarray) and TY.dtype == np.dtype(str(dtype)[6:])
    tol = 1e-3 if dtype == torch.float32 else 1e-8
    np.testing.assert_allclose(B, B_true, atol=tol)
    np.testing.assert_allclose(t, t_true, atol=tol * 10)
    np.testing.assert_allclose(TY, X, atol=tol * 10)


def test_fish_with_shear(fish):
    X, _ = fish
    shear = np.array([[1.0, 0.5], [0.0, 1.0]])
    A = (
        np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]])
        @ shear
    )
    Y = X @ A + np.array([0.5, 1.0])
    TY, _ = AffineRegistration(X, Y).register()
    assert np.abs(TY - X).max() < 1e-6


def test_initial_parameters_and_new_points(fish):
    X, Y = fish
    B0 = np.array([[1.1, 0.2], [0.0, 0.9]])
    reg = AffineRegistration(X, Y, B=B0, t=[0.3, 0.4], max_iterations=0)
    TY, (_B, _t) = reg.register()
    np.testing.assert_allclose(TY, Y @ B0 + [0.3, 0.4], atol=1e-12)
    np.testing.assert_allclose(reg.transform_point_cloud(Y[:3]), TY[:3], atol=1e-12)


def test_objective_decreases_and_converges(fish):
    X, Y = fish
    errors = []
    reg = AffineRegistration(lift_to_3d(X), lift_to_3d(Y))
    reg.register(callback=lambda **kw: errors.append(kw["error"]))
    assert all(isinstance(e, float) for e in errors)
    assert errors[-1] < errors[0]
    assert bool(reg.converged)
    assert numpy(reg.diff) <= reg.tolerance


def test_invalid_initial_matrix(fish):
    X, Y = fish
    with pytest.raises(ValueError, match="shape"):
        AffineRegistration(X, Y, B=np.eye(3))
    with pytest.raises(ValueError, match="NaN"):
        AffineRegistration(X, Y, B=np.full((2, 2), np.nan))
