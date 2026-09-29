"""Rigid registration of the Stanford bunny, rotated by 45 degrees and shifted."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import RigidRegistration, datasets


def main():
    args = parse_args(__doc__)
    X, _ = datasets.load_bunny()
    angle = np.radians(45)
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    center = X.mean(0)
    Y = (X - center) @ R + center + np.array([0.06, -0.03, 0.02])  # rotate about the center

    animate = Animator(args, X, Y, title="rigid", view=(90, -90))  # front view
    reg = RigidRegistration(X, Y, device=args.device, dtype=args.dtype)
    TY, (s, R_est, t) = reg.register(callback=animate)
    print(f"{reg.iteration} iterations, scale {float(s):.4f}")
    print(f"max alignment error: {np.abs(TY - X).max():.1e}")
    animate.finish()


if __name__ == "__main__":
    main()
