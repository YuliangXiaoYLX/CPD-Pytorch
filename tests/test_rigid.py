from __future__ import annotations

import numpy as np
import pytest
import torch

from cpd_pytorch import RigidRegistration
from helpers import DEVICE_DTYPES, max_rel_error, numpy, rotation

# float32 cannot resolve residuals below ~1e-7 of the data scale.
TOL = {torch.float32: 1e-4, torch.float64: 1e-9}


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
@pytest.mark.parametrize(
    ("angles", "scale"),
    [((0.5,), 1.0), ((0.5,), 1.7), ((0.3, -0.4, 0.2), 1.0), ((0.3, -0.4, 0.2), 0.6)],
)
def test_recovers_similarity_transform(device, dtype, angles, scale, rng, fish, bunny):
    # Real, asymmetric shapes: an isotropic random blob has no identifiable rotation.
    shape = fish[0] if len(angles) == 1 else bunny[0]
    X = 5.0 * shape + 20.0  # far from the origin, to exercise float32
    D = X.shape[1]
    R_true = rotation(*angles)
    # Rotate about the centroid and shift a little, so the start overlaps the target.
    center = X.mean(0)
    Y = (X - center) @ R_true.T / scale + center + 0.3 * rng.normal(size=D)
    t_true = (X - scale * Y @ R_true).mean(0)  # X = scale * Y @ R_true + t_true
    X_t = torch.as_tensor(X, dtype=dtype, device=device)
    Y_t = torch.as_tensor(Y, dtype=dtype, device=device)

    reg = RigidRegistration(X_t, Y_t)
    TY, (s, R, t) = reg.register()

    assert TY.device.type == device and TY.dtype == dtype
    assert R.dtype == dtype and s.shape == () and t.shape == (D,)
    tol = TOL[dtype]
    assert abs(float(s) - scale) < tol * 10
    assert np.abs(numpy(R) - R_true).max() < tol * 10
    assert np.abs(numpy(t) - t_true).max() < tol * 100
    assert np.abs(numpy(TY) - X).max() < tol * 100
    assert reg.converged


def test_no_scale_keeps_unit_scale(fish):
    X, Y = fish
    _, (s, R, _) = RigidRegistration(X, 2.0 * Y, scale=False).register()
    assert float(s) == 1.0
    np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_any_dimension(rng):
    X = rng.normal(size=(300, 5)) * np.array([5.0, 3.0, 2.0, 1.0, 0.5])  # anisotropic
    A = 0.2 * rng.normal(size=(5, 5))
    A = A - A.T  # skew-symmetric, so the Cayley transform below is a rotation
    Q = np.linalg.solve(np.eye(5) - A, np.eye(5) + A)
    TY, (_, R, _) = RigidRegistration(X, X @ Q.T).register()
    assert np.abs(TY - X).max() < 1e-8
    np.testing.assert_allclose(R, Q, atol=1e-8)


def test_initial_parameters_numpy_and_tensor(fish):
    X, Y = fish
    R0 = rotation(0.2)
    # A float32 tensor is converted to the registration dtype (float64 here), keeping
    # float32 precision.
    for R_init, atol in ((R0, 1e-12), (torch.as_tensor(R0, dtype=torch.float32), 1e-6)):
        reg = RigidRegistration(X, Y, R=R_init, t=np.array([[0.1, -0.1]]), s=1.5, max_iterations=0)
        assert reg.R.dtype == torch.float64
        TY, (s, _R, t) = reg.register()
        np.testing.assert_allclose(TY, 1.5 * Y @ R0 + np.array([0.1, -0.1]), atol=atol)
        np.testing.assert_allclose(t, [0.1, -0.1], atol=1e-12)
        assert float(s) == pytest.approx(1.5)


def test_initial_rotation_of_90_degrees_is_valid(fish):
    """pycpd rejected valid rotations because it tested for positive definiteness."""
    X, Y = fish
    RigidRegistration(X, Y, R=rotation(np.pi / 2))
    RigidRegistration(X, Y, R=rotation(np.pi))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"R": np.eye(3)}, "shape"),
        ({"R": 2 * np.eye(2)}, "rotation"),
        ({"R": np.diag([1.0, -1.0])}, "rotation"),
        ({"t": np.zeros(3)}, "shape"),
        ({"s": -1.0}, "positive"),
    ],
)
def test_invalid_initial_parameters(fish, kwargs, match):
    X, Y = fish
    with pytest.raises(ValueError, match=match):
        RigidRegistration(X, Y, **kwargs)


def test_transform_point_cloud_new_points(bunny):
    X, Y = bunny
    reg = RigidRegistration(X, Y)
    TY, _ = reg.register()
    moved = reg.transform_point_cloud(Y)
    assert isinstance(moved, np.ndarray)
    np.testing.assert_allclose(moved, TY, atol=1e-12)
    moved_t = reg.transform_point_cloud(torch.as_tensor(Y[:5]))
    assert isinstance(moved_t, torch.Tensor)
    assert max_rel_error(moved_t, TY[:5]) < 1e-12


def test_exact_fit_stops_early(fish):
    """With noise-free data sigma2 collapses; iteration must stop instead of cycling."""
    X, _ = fish
    for dtype in (torch.float64, torch.float32):
        reg = RigidRegistration(X, X @ rotation(0.3) + 1.0, dtype=dtype)
        TY, _ = reg.register()
        assert reg.iteration < reg.max_iterations
        assert bool(reg.converged)
        assert np.abs(numpy(TY) - X).max() < TOL[dtype] * 10
