from __future__ import annotations

import numpy as np
import pytest
import torch

from cpd_pytorch import ConstrainedDeformableRegistration, DeformableRegistration
from helpers import DEVICE_DTYPES, max_rel_error, numpy

IDS = np.array([1, 10, 20, 30])


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
def test_landmarks_are_matched(device, dtype, fish):
    """Regression test: float64 used to crash with a float/double dtype mismatch."""
    X, Y = fish
    X = X[:61]  # the target is missing part of the shape
    reg = ConstrainedDeformableRegistration(
        X, Y, source_id=IDS, target_id=IDS, device=device, dtype=dtype
    )
    TY, (_G, _W) = reg.register()
    assert TY.dtype == np.dtype(str(dtype)[6:])
    assert np.abs(TY[IDS] - X[IDS]).max() < 1e-2


def test_priors_pull_toward_landmarks(fish):
    X, Y = fish
    X = X[:61]
    free, _ = DeformableRegistration(X, Y).register()
    guided, _ = ConstrainedDeformableRegistration(X, Y, source_id=IDS, target_id=IDS).register()
    assert np.abs(guided[IDS] - X[IDS]).max() < np.abs(free[IDS] - X[IDS]).max()


def test_duplicate_pairs_count_once(fish):
    X, Y = fish
    once = ConstrainedDeformableRegistration(X, Y, source_id=IDS, target_id=IDS, max_iterations=5)
    twice = ConstrainedDeformableRegistration(
        X, Y, source_id=np.r_[IDS, IDS], target_id=np.r_[IDS, IDS], max_iterations=5
    )
    assert max_rel_error(twice.register()[0], once.register()[0]) < 1e-12


def test_ids_as_lists_and_tensors(fish):
    X, Y = fish
    for ids in (IDS.tolist(), torch.as_tensor(IDS)):
        reg = ConstrainedDeformableRegistration(
            X, Y, source_id=ids, target_id=ids, low_rank=True, num_eig=20
        )
        TY, _ = reg.register()
        assert np.isfinite(numpy(TY)).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "required"),
        ({"source_id": IDS, "target_id": IDS[:2]}, "same length"),
        ({"source_id": IDS.astype(float), "target_id": IDS}, "integer"),
        ({"source_id": np.array([[1, 2]]), "target_id": IDS}, "1D"),
        ({"source_id": np.array([1, 999]), "target_id": np.array([1, 2])}, r"\[0, 91\)"),
        ({"source_id": IDS, "target_id": IDS, "e_alpha": 0}, "e_alpha"),
    ],
)
def test_invalid_correspondences(fish, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ConstrainedDeformableRegistration(*fish, **kwargs)
