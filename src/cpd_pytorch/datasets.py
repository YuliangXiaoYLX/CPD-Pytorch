"""Small example point sets shipped with the package."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_bunny", "load_fish"]

_DATA = Path(__file__).resolve().parent / "data"


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.loadtxt(_DATA / f"{name}_target.txt"), np.loadtxt(_DATA / f"{name}_source.txt")


def load_fish() -> tuple[np.ndarray, np.ndarray]:
    """2D fish outlines, the classic example of Myronenko & Song (2010).

    Returns
    -------
    X : numpy.ndarray, shape (91, 2)
        Target points.
    Y : numpy.ndarray, shape (91, 2)
        Source points: a non-rigidly deformed fish.
    """
    return _load("fish")


def load_bunny() -> tuple[np.ndarray, np.ndarray]:
    """3D Stanford bunny, subsampled.

    Returns
    -------
    X : numpy.ndarray, shape (453, 3)
        Target points.
    Y : numpy.ndarray, shape (453, 3)
        Source points: the target translated by (1, 1, 1).
    """
    return _load("bunny")
