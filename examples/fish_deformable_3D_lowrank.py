"""Low-rank deformable registration: approximates the kernel by its leading eigenpairs."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import DeformableRegistration, datasets


def lift_to_3d(points):
    return np.vstack([np.c_[points, np.zeros(len(points))], np.c_[points, np.ones(len(points))]])


def main():
    args = parse_args(__doc__)
    X, Y = (lift_to_3d(points) for points in datasets.load_fish())

    animate = Animator(args, X, Y, title="low-rank deformable")
    reg = DeformableRegistration(
        X, Y, low_rank=True, num_eig=40, device=args.device, dtype=args.dtype
    )
    reg.register(callback=animate)
    print(f"{reg.iteration} iterations, kept eigenvalues {reg.S[0]:.3g} ... {reg.S[-1]:.3g}")
    animate.finish()


if __name__ == "__main__":
    main()
