"""Affine registration of two different 3D fish (stacked 2D outlines)."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import AffineRegistration, datasets


def lift_to_3d(points):
    return np.vstack([np.c_[points, np.zeros(len(points))], np.c_[points, np.ones(len(points))]])


def main():
    args = parse_args(__doc__)
    X, Y = (lift_to_3d(points) for points in datasets.load_fish())

    animate = Animator(args, X, Y, title="affine 3D")
    reg = AffineRegistration(X, Y, device=args.device, dtype=args.dtype)
    reg.register(callback=animate)
    print(f"{reg.iteration} iterations, sigma2 = {float(reg.sigma2):.3e}")
    animate.finish()


if __name__ == "__main__":
    main()
