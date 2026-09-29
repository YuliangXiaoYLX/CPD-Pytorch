"""Constrained deformable registration of a partial 3D fish with known correspondences."""

import numpy as np

from _plot import Animator, parse_args
from cpd_pytorch import ConstrainedDeformableRegistration, datasets

LANDMARKS = [1, 10, 20, 30]


def lift_to_3d(points):
    return np.vstack([np.c_[points, np.zeros(len(points))], np.c_[points, np.ones(len(points))]])


def main():
    args = parse_args(__doc__)
    fish_X, fish_Y = datasets.load_fish()
    fish_X = fish_X[:61]  # the target is missing part of the shape
    X, Y = lift_to_3d(fish_X), lift_to_3d(fish_Y)
    # The same landmarks on both layers of the stacked shapes.
    source_id = np.array(LANDMARKS + [len(fish_Y) + i for i in LANDMARKS])
    target_id = np.array(LANDMARKS + [len(fish_X) + i for i in LANDMARKS])

    animate = Animator(args, X, Y, title="constrained 3D", highlight=source_id)
    reg = ConstrainedDeformableRegistration(
        X, Y, source_id=source_id, target_id=target_id, device=args.device, dtype=args.dtype
    )
    TY, _ = reg.register(callback=animate)
    print(f"landmark error: {np.abs(TY[source_id] - X[target_id]).max():.2e}")
    animate.finish()


if __name__ == "__main__":
    main()
