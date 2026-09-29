"""Numerical parity with pycpd, the NumPy implementation this package ports.

pycpd (master, pinned in pyproject.toml) is the reference. In float64 every state of the
EM iteration must match it to rounding error. The only intended differences are the fixed
objectives: the affine ``q`` (pycpd uses tr(B YPY B) instead of tr(Bᵀ YPY B)) and the
deformable ``q`` (pycpd reports inf).
"""

from __future__ import annotations

import numpy as np
import pytest

import cpd_pytorch as cpd
from helpers import lift_to_3d, max_rel_error

pycpd = pytest.importorskip("pycpd")
if not hasattr(pycpd, "ConstrainedDeformableRegistration"):
    pytest.skip("needs pycpd master (see the test dependency group)", allow_module_level=True)

IDS = np.array([1, 10, 20, 30])


def _cases(fish, bunny):
    X2, Y2 = fish
    Xb, Yb = bunny
    noisy = Yb + np.random.default_rng(0).normal(0, 0.01, Yb.shape)
    return {
        "rigid": ("RigidRegistration", {"X": X2, "Y": Y2}),
        "rigid-no-scale": ("RigidRegistration", {"X": X2, "Y": Y2, "scale": False}),
        "rigid-outliers-3d": ("RigidRegistration", {"X": Xb, "Y": noisy, "w": 0.2}),
        "affine": ("AffineRegistration", {"X": X2, "Y": Y2}),
        "affine-outliers-3d": (
            "AffineRegistration",
            {"X": lift_to_3d(X2), "Y": lift_to_3d(Y2), "w": 0.1},
        ),
        "deformable": ("DeformableRegistration", {"X": X2, "Y": Y2}),
        "deformable-params-3d": (
            "DeformableRegistration",
            {"X": lift_to_3d(X2), "Y": lift_to_3d(Y2), "alpha": 1.0, "beta": 0.5, "w": 0.05},
        ),
        "deformable-low-rank": (
            "DeformableRegistration",
            {"X": lift_to_3d(X2), "Y": lift_to_3d(Y2), "low_rank": True, "num_eig": 30},
        ),
        "constrained": (
            "ConstrainedDeformableRegistration",
            {"X": X2[:61], "Y": Y2, "source_id": IDS, "target_id": IDS},
        ),
    }


CASES = [
    "rigid",
    "rigid-no-scale",
    "rigid-outliers-3d",
    "affine",
    "affine-outliers-3d",
    "deformable",
    "deformable-params-3d",
    "deformable-low-rank",
    "constrained",
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("iterations", [1, 25])
def test_matches_pycpd(case, iterations, fish, bunny):
    name, kwargs = _cases(fish, bunny)[case]
    ref = getattr(pycpd, name)(**kwargs, max_iterations=iterations, tolerance=0.0)
    new = getattr(cpd, name)(**kwargs, max_iterations=iterations, tolerance=0.0)
    TY_ref, params_ref = ref.register()
    TY_new, params_new = new.register()

    assert isinstance(TY_new, np.ndarray)
    assert max_rel_error(TY_new, TY_ref) < 1e-8
    assert max_rel_error(new.sigma2, ref.sigma2) < 1e-8
    for actual, expected in zip(params_new, params_ref):
        assert max_rel_error(actual, expected) < 1e-8
    # Posterior and E-step statistics of the last iteration.
    assert max_rel_error(new.P, ref.P) < 1e-8
    assert max_rel_error(new.P1, ref.P1) < 1e-8
    assert max_rel_error(new.Pt1, ref.Pt1) < 1e-8
    if name in ("RigidRegistration",):
        assert max_rel_error(new.q, ref.q) < 1e-8


@pytest.mark.parametrize("case", ["rigid", "rigid-outliers-3d", "deformable", "constrained"])
def test_same_number_of_iterations_as_pycpd(case, fish, bunny):
    name, kwargs = _cases(fish, bunny)[case]
    ref = getattr(pycpd, name)(**kwargs)
    new = getattr(cpd, name)(**kwargs)
    ref.register()
    new.register()
    assert new.iteration == ref.iteration


def test_affine_objective_is_the_negative_log_likelihood_bound(fish):
    """q must equal the EM objective sum_mn P_mn |x_n - T(y_m)|² / 2σ² + (Np D / 2) log σ²."""
    X, Y = fish
    reg = cpd.AffineRegistration(X, Y, max_iterations=3, tolerance=0.0)
    reg.transform_point_cloud()
    for _ in range(3):
        sigma2 = reg.sigma2.clone()
        reg.expectation()
        reg.update_transform()
        reg.transform_point_cloud()
        P = reg.P
        residual = (P * ((reg.TY[:, None, :] - reg.X[None, :, :]) ** 2).sum(-1)).sum()
        expected = residual / (2 * sigma2) + reg.D * reg.Np / 2 * np.log(float(sigma2))
        reg.update_variance()
        assert max_rel_error(reg.q, expected) < 1e-10
