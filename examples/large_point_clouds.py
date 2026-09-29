"""Register large point clouds with bounded memory, then extract correspondences.

The E-step never stores the full M x N posterior: target points are processed in blocks
of ``chunk_size``, so memory grows with ``M * chunk_size`` rather than ``M * N``. Use a GPU
and float32 for the largest problems, e.g.

    python large_point_clouds.py --points 200000 --device cuda --dtype float32
"""

import time

import numpy as np

from _plot import parse_args
from cpd_pytorch import RigidRegistration, datasets


def dense_bunny(points, rng):
    """Resample the bunny with jitter to get an arbitrary number of surface points."""
    bunny, _ = datasets.load_bunny()
    picks = rng.integers(len(bunny), size=points)
    return bunny[picks] + rng.normal(0, 0.002, (points, 3))


def main():
    args = parse_args(
        __doc__, lambda parser: parser.add_argument("--points", type=int, default=20_000)
    )
    points = args.points
    rng = np.random.default_rng(0)
    X = dense_bunny(points, rng)
    angle = 0.4
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    Y = dense_bunny(points, rng) @ R + np.array([0.05, -0.02, 0.03])
    # 10% outliers in the target: raise w accordingly.
    X[: points // 10] = rng.uniform(X.min(0), X.max(0), (points // 10, 3))

    start = time.perf_counter()
    reg = RigidRegistration(X, Y, w=0.1, chunk_size=2048, device=args.device, dtype=args.dtype)
    TY, (s, R_est, t) = reg.register()
    seconds = time.perf_counter() - start
    print(f"{points:,} x {points:,} points: {reg.iteration} iterations in {seconds:.1f} s")
    print(f"rotation error: {np.abs(R_est - R.T).max():.2e}")

    # Most probable source point for every target point: X[n] <-> Y[index[n]].
    index = reg.correspondences()
    distance = np.linalg.norm(X - TY[index], axis=1)
    # Pt1[n]: probability that target point n is explained by the source rather than by
    # the outlier distribution (from the last E-step).
    inlier_probability = reg.Pt1.cpu().numpy()
    outlier = inlier_probability < 0.5
    planted = np.arange(points) < points // 10
    median = np.median(distance[~outlier])
    print(f"median distance to the matched source point (inliers): {median:.4f}")
    print(
        f"flagged {outlier.mean():.1%} of target points as outliers, including "
        f"{outlier[planted].mean():.0%} of the planted ones (the rest landed near the surface)"
    )


if __name__ == "__main__":
    main()
