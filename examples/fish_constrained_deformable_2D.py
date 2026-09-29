"""Deformable registration of a partial 2D fish, guided by four known correspondences."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import ConstrainedDeformableRegistration, datasets

LANDMARKS = np.array([1, 10, 20, 30])


def main():
    args = parse_args(__doc__)
    X, Y = datasets.load_fish()
    X = X[:61]  # the target is missing part of the shape

    animate = Animator(args, X, Y, title="constrained", highlight=LANDMARKS)
    # Source point LANDMARKS[i] must land on target point LANDMARKS[i].
    # e_alpha: 1e-8 trusts the correspondences fully, 1 barely uses them.
    reg = ConstrainedDeformableRegistration(
        X, Y, source_id=LANDMARKS, target_id=LANDMARKS, e_alpha=1e-8,
        device=args.device, dtype=args.dtype,
    )  # fmt: skip
    TY, _ = reg.register(callback=animate)
    print(f"landmark error: {np.abs(TY[LANDMARKS] - X[LANDMARKS]).max():.2e}")
    animate.finish()


if __name__ == "__main__":
    main()
