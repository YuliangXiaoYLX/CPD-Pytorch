"""Block-wise E-step, correspondences, the posterior matrix and normalization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import cpd_pytorch as cpd
from cpd_pytorch.emregistration import initialize_sigma2
from helpers import lift_to_3d, max_rel_error, rotation


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("RigidRegistration", {"w": 0.1}),
        ("AffineRegistration", {}),
        ("DeformableRegistration", {}),
        ("DeformableRegistration", {"low_rank": True, "num_eig": 20}),
    ],
)
def test_chunked_estep_equals_single_block(name, kwargs, fish):
    X, Y = lift_to_3d(fish[0]), lift_to_3d(fish[1])
    cls = getattr(cpd, name)
    TY_full, _ = cls(X, Y, **kwargs).register()
    reg = cls(X, Y, chunk_size=7, **kwargs)  # 26 blocks of target points
    TY_chunked, _ = reg.register()
    assert max_rel_error(TY_chunked, TY_full) < 1e-10


def test_P_matches_statistics(fish):
    reg = cpd.RigidRegistration(*fish, w=0.2, chunk_size=10, max_iterations=5)
    reg.register()
    P = reg.P
    assert P.shape == (91, 91)
    assert max_rel_error(P.sum(1), reg.P1) < 1e-12
    assert max_rel_error(P.sum(0), reg.Pt1) < 1e-12
    assert max_rel_error(P.sum(), reg.Np) < 1e-12
    assert cpd.RigidRegistration(*fish).P.abs().sum() == 0  # before any E-step


def test_initialize_sigma2_matches_definition(rng):
    X = torch.as_tensor(rng.normal(size=(40, 3)) + 100)
    Y = torch.as_tensor(rng.normal(size=(30, 3)))
    expected = ((X[None] - Y[:, None]) ** 2).sum() / (3 * 40 * 30)
    assert max_rel_error(initialize_sigma2(X, Y), expected) < 1e-12


def test_correspondences_recover_permutation(bunny, rng):
    X, _ = bunny
    perm = rng.permutation(len(X))
    Y = X[perm] @ rotation(0.2, 0.1, -0.1) + 0.5
    reg = cpd.RigidRegistration(X, Y, chunk_size=50)
    reg.register()
    index, probability = reg.correspondences(return_probability=True)
    # X[n] corresponds to Y[index[n]], i.e. index = inverse permutation.
    np.testing.assert_array_equal(index, np.argsort(perm))
    assert probability.min() > 0.99


def test_correspondences_batched_tensor(fish):
    X, Y = (torch.as_tensor(a) for a in fish)
    reg = cpd.DeformableRegistration(torch.stack([X, X]), torch.stack([Y, Y]))
    reg.register()
    index = reg.correspondences()
    assert isinstance(index, torch.Tensor) and index.shape == (2, 91) and index.dtype == torch.long
    assert torch.equal(index[0], index[1])


@pytest.mark.parametrize("units", [1e-3, 1.0, 1e3])
def test_normalize_makes_deformable_scale_invariant(units, fish):
    X, Y = fish
    ref = cpd.DeformableRegistration(X, Y, normalize=True)
    TY_ref, _ = ref.register()
    reg = cpd.DeformableRegistration(units * X + 7.0, units * Y + 7.0, normalize=True)
    TY, _ = reg.register()
    assert reg.iteration == ref.iteration
    assert max_rel_error((TY - 7.0) / units, TY_ref) < 1e-8
    assert float(reg.sigma2) == pytest.approx(units**2 * float(ref.sigma2), rel=1e-8)
    np.testing.assert_allclose(reg.transform_point_cloud(units * Y + 7.0), TY, atol=1e-8 * units)


def test_normalize_keeps_rigid_transform_rigid(fish):
    """A shared scale factor keeps scale=False rigid (MATLAB CPD's separate scaling does not)."""
    X, Y = fish
    TY, (s, R, t) = cpd.RigidRegistration(10 * X, Y, scale=False, normalize=True).register()
    assert float(s) == 1.0
    np.testing.assert_allclose(TY, Y @ R + t, atol=1e-10)


def _centered(points: np.ndarray) -> np.ndarray:
    return points - points.mean(0)


@pytest.mark.parametrize("name", ["RigidRegistration", "AffineRegistration"])
def test_normalize_does_not_change_rigid_affine_solution(name, fish):
    """Without outliers, rigid and affine CPD are equivariant under a shared rescaling.

    With both sets pre-centered, normalize=True differs from normalize=False only by that
    rescaling, so the results must agree to rounding error.
    """
    X, Y = _centered(fish[0]), _centered(fish[1])
    cls = getattr(cpd, name)
    TY_plain, params_plain = cls(X, Y).register()
    TY_norm, params_norm = cls(X, Y, normalize=True).register()
    assert max_rel_error(TY_norm, TY_plain) < 1e-9
    for a, b in zip(params_norm, params_plain):
        assert np.abs(a - b).max() < 1e-9


def test_sigma2_in_data_units(fish):
    X, Y = _centered(fish[0]), _centered(fish[1])
    plain = cpd.RigidRegistration(X, Y, max_iterations=0)
    scaled = cpd.RigidRegistration(3 * X, 3 * Y, max_iterations=0, normalize=True)
    assert float(scaled.sigma2) == pytest.approx(9 * float(plain.sigma2))
    reg = cpd.RigidRegistration(X, Y, sigma2=0.5, normalize=True)
    assert float(reg.sigma2) == pytest.approx(0.5)
