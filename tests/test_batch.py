"""Batched registration must equal registering each pair on its own."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import cpd_pytorch as cpd
from helpers import lift_to_3d, max_rel_error, rotation


def _batch(fish, rng, B=3):
    X, Y = fish
    Xs = np.stack([X @ rotation(0.1 * b) + 0.2 * b for b in range(B)])
    Ys = np.stack([Y + rng.normal(0, 0.01 * (b + 1), Y.shape) for b in range(B)])
    return Xs, Ys


METHODS = [
    ("RigidRegistration", {}),
    ("RigidRegistration", {"scale": False, "w": 0.1}),
    ("AffineRegistration", {}),
    ("DeformableRegistration", {}),
    ("DeformableRegistration", {"low_rank": True, "num_eig": 15}),
    ("ConstrainedDeformableRegistration", {"source_id": [1, 10, 20], "target_id": [1, 10, 20]}),
]


@pytest.mark.parametrize(("name", "kwargs"), METHODS)
@pytest.mark.parametrize("normalize", [False, True])
def test_batch_equals_loop(name, kwargs, normalize, fish, rng):
    Xs, Ys = _batch(fish, rng)
    cls = getattr(cpd, name)
    batched = cls(Xs, Ys, normalize=normalize, **kwargs)
    TY, params = batched.register()
    iterations = []
    for b in range(len(Xs)):
        single = cls(Xs[b], Ys[b], normalize=normalize, **kwargs)
        TY_b, params_b = single.register()
        iterations.append(single.iteration)
        assert max_rel_error(TY[b], TY_b) < 1e-9
        for p, p_b in zip(params, params_b):
            assert max_rel_error(p[b], p_b) < 1e-8
        assert max_rel_error(batched.sigma2[b], single.sigma2) < 1e-8
    # Pairs converged at different iterations and were frozen at their own optimum.
    assert batched.iteration == max(iterations)
    assert batched.converged.shape == (len(Xs),)


def test_shared_target_is_broadcast(fish, rng):
    X, Y = fish
    Ys = np.stack([Y, Y @ rotation(0.2), 1.5 * Y])
    reg = cpd.RigidRegistration(X, Ys)
    TY, (s, R, t) = reg.register()
    assert TY.shape == (3, 91, 2) and R.shape == (3, 2, 2) and s.shape == (3,) and t.shape == (3, 2)
    for b in range(3):
        TY_b, _ = cpd.RigidRegistration(X, Ys[b]).register()
        assert max_rel_error(TY[b], TY_b) < 1e-9


def test_batched_initial_parameters_and_new_points(fish):
    X, Y = fish
    Xs = np.stack([X, X])
    Ys = np.stack([Y, Y])
    R0 = np.stack([rotation(0.1), rotation(-0.1)])
    reg = cpd.RigidRegistration(Xs, Ys, R=R0, s=np.array([1.0, 2.0]), max_iterations=0)
    TY, (_s, _R, _t) = reg.register()
    np.testing.assert_allclose(TY[1], 2.0 * Y @ rotation(-0.1), atol=1e-12)
    moved = reg.transform_point_cloud(Y[:4])  # shared points, batched transform
    assert moved.shape == (2, 4, 2)
    np.testing.assert_allclose(moved[0], Y[:4] @ rotation(0.1), atol=1e-12)


def test_batched_callback_errors(fish, rng):
    Xs, Ys = _batch(fish, rng)
    errors = []
    cpd.RigidRegistration(Xs, Ys, max_iterations=3).register(
        lambda **kw: errors.append(kw["error"])
    )
    assert len(errors) == 3 and all(e.shape == (3,) for e in errors)


def test_batched_tensors_on_float32(fish, rng):
    Xs, Ys = (torch.as_tensor(a, dtype=torch.float32) for a in _batch(fish, rng))
    TY, _ = cpd.DeformableRegistration(lift_to_3d_batch(Xs), lift_to_3d_batch(Ys)).register()
    assert TY.shape == (3, 182, 3) and TY.dtype == torch.float32


def lift_to_3d_batch(points: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.as_tensor(lift_to_3d(p.numpy()), dtype=points.dtype) for p in points])
