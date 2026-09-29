"""Rigid registration of a rotated and shifted 2D fish."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import RigidRegistration, datasets


def main():
    args = parse_args(__doc__)
    X, _ = datasets.load_fish()
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Y = X @ R + np.array([0.5, 1.0])  # a rigidly moved copy of the target

    animate = Animator(args, X, Y, title="rigid")
    reg = RigidRegistration(X, Y, device=args.device, dtype=args.dtype)
    TY, (s, R, t) = reg.register(callback=animate)
    print(f"{reg.iteration} iterations, scale {float(s):.4f}, translation {t}")
    print(f"max alignment error: {np.abs(TY - X).max():.2e}")
    animate.finish()


if __name__ == "__main__":
    main()
