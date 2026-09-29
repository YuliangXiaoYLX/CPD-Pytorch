from __future__ import annotations

import numpy as np
import pytest
import torch

from cpd_pytorch import DeformableRegistration, gaussian_kernel
from cpd_pytorch.utils import low_rank_eigen
from helpers import DEVICE_DTYPES, lift_to_3d, max_rel_error, numpy


def _mean_nearest_distance(A: np.ndarray, B: np.ndarray) -> float:
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d.min(1)).mean())


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
def test_fish_is_aligned(device, dtype, fish):
    X, Y = fish
    reg = DeformableRegistration(
        torch.as_tensor(X, device=device, dtype=dtype),
        torch.as_tensor(Y, device=device, dtype=dtype),
    )
    TY, (G, W) = reg.register()
    assert TY.dtype == dtype and G.shape == (91, 91) and W.shape == (91, 2)
    before, after = _mean_nearest_distance(Y, X), _mean_nearest_distance(numpy(TY), X)
    assert after < 0.2 * before  # 0.214 -> 0.033, as with pycpd
    assert np.isfinite(numpy(reg.q))
    # TY = Y + G W
    assert max_rel_error(reg.Y + G @ W, TY) < (1e-4 if dtype == torch.float32 else 1e-10)


def test_transform_new_points_matches_TY(fish):
    X, Y = fish
    reg = DeformableRegistration(X, Y)
    TY, _ = reg.register()
    np.testing.assert_allclose(reg.transform_point_cloud(Y), TY, atol=1e-10)


def test_objective_includes_regularizer(fish):
    X, Y = fish
    reg = DeformableRegistration(X, Y, max_iterations=5)
    reg.register()
    G, W = reg.G, reg.W
    regularizer = 0.5 * reg.alpha * float((W * (G @ W)).sum())
    assert regularizer > 0
    assert np.isfinite(float(reg.q))


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
def test_low_rank_eigen_matches_dense(device, dtype, rng):
    Y = torch.as_tensor(rng.normal(size=(800, 3)), dtype=dtype, device=device)
    k = 40
    Q, S = low_rank_eigen(Y, beta=1.5, num_eig=k)  # 800 > 4 * (k + 20): randomized path
    assert Q.device.type == device
    Q, S, Y = Q.cpu().double(), S.cpu().double(), Y.cpu().double()
    S_all, Q_all = torch.linalg.eigh(gaussian_kernel(Y, 1.5))
    S_all, Q_all = S_all.flip(-1), Q_all.flip(-1)
    best = (Q_all[:, :k] * S_all[:k]) @ Q_all[:, :k].T  # optimal rank-k approximation
    truncation = float(S_all[k])  # its spectral-norm error
    error = float(torch.linalg.matrix_norm((Q * S) @ Q.T - best, ord=2))
    # float64: far below the truncation error; float32: at its rounding level.
    assert error < (1e-4 * truncation if dtype == torch.float64 else 1e-3 * float(S_all[0]))
    assert max_rel_error(S, S_all[:k]) < (1e-8 if dtype == torch.float64 else 1e-4)
    assert max_rel_error(Q.T @ Q, torch.eye(k, dtype=torch.float64)) < 1e-4


def test_low_rank_eigen_matrix_free_equals_dense(rng):
    Y = torch.as_tensor(rng.normal(size=(600, 2)))
    Q1, S1 = low_rank_eigen(Y, 1.0, 30)
    Q2, S2 = low_rank_eigen(Y, 1.0, 30, max_elements=10_000)  # G never formed
    assert max_rel_error(S2, S1) < 1e-8
    assert max_rel_error((Q2 * S2) @ Q2.T, (Q1 * S1) @ Q1.T) < 1e-8


def test_low_rank_close_to_full(fish):
    X, Y = lift_to_3d(fish[0]), lift_to_3d(fish[1])
    TY_full, _ = DeformableRegistration(X, Y).register()
    TY_low, _ = DeformableRegistration(X, Y, low_rank=True, num_eig=40).register()
    assert np.abs(TY_low - TY_full).max() < 1e-6


def test_low_rank_does_not_form_huge_kernel(rng):
    """A low-rank model returns G=None (with a warning) instead of allocating M x M."""
    import cpd_pytorch.deformable_registration as module

    X = rng.normal(size=(300, 2))
    reg = DeformableRegistration(X, X + 0.01, low_rank=True, num_eig=20, max_iterations=2)
    original = module._MAX_G_ELEMENTS
    module._MAX_G_ELEMENTS = 1000
    try:
        with pytest.warns(RuntimeWarning, match="kernel matrix"):
            _, (G, W) = reg.register()
    finally:
        module._MAX_G_ELEMENTS = original
    assert G is None and W.shape == (300, 2)
    assert reg.G.shape == (300, 300)  # explicit access still works


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"alpha": 0}, "alpha"), ({"beta": -1}, "beta"), ({"num_eig": 0}, "num_eig")],
)
def test_invalid_parameters(fish, kwargs, match):
    with pytest.raises(ValueError, match=match):
        DeformableRegistration(*fish, **kwargs)
