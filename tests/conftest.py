from __future__ import annotations

import numpy as np
import pytest

from cpd_pytorch import datasets


@pytest.fixture(scope="session")
def fish() -> tuple[np.ndarray, np.ndarray]:
    return datasets.load_fish()


@pytest.fixture(scope="session")
def bunny() -> tuple[np.ndarray, np.ndarray]:
    return datasets.load_bunny()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
