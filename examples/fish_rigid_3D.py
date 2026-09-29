"""Rigid registration of a rotated and shifted 3D fish (two stacked 2D outlines)."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import RigidRegistration, datasets


def main():
    args = parse_args(__doc__)
    fish, _ = datasets.load_fish()
    X = np.vstack([np.c_[fish, np.zeros(len(fish))], np.c_[fish, np.ones(len(fish))]])
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Y = X @ R + np.array([0.5, 1.0, 0.0])

    animate = Animator(args, X, Y, title="rigid 3D")
    reg = RigidRegistration(X, Y, device=args.device, dtype=args.dtype)
    TY, _ = reg.register(callback=animate)
    print(f"{reg.iteration} iterations, max alignment error {np.abs(TY - X).max():.2e}")
    animate.finish()


if __name__ == "__main__":
    main()
