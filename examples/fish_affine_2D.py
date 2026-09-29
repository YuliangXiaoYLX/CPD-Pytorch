"""Affine registration of a rotated, sheared and shifted 2D fish."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import AffineRegistration, datasets


def main():
    args = parse_args(__doc__)
    X, _ = datasets.load_fish()
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shear = np.array([[1.0, 0.5], [0.0, 1.0]])
    Y = X @ (R @ shear) + np.array([0.5, 1.0])

    animate = Animator(args, X, Y, title="affine")
    reg = AffineRegistration(X, Y, device=args.device, dtype=args.dtype)
    TY, (B, t) = reg.register(callback=animate)
    print(f"{reg.iteration} iterations, max alignment error {np.abs(TY - X).max():.2e}")
    animate.finish()


if __name__ == "__main__":
    main()
