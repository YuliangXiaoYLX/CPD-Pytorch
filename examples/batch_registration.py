"""Register many point-set pairs in one call and compare with a Python loop.

A batch is a leading dimension: X of shape (B, N, D) and Y of shape (B, M, D). An unbatched
X (or Y) is shared by all pairs. Each pair keeps its own parameters and stops at its own
convergence, so the results equal registering the pairs one by one.
"""

import time

import numpy as np

from _plot import parse_args
from cpd_pytorch import RigidRegistration, datasets


def main():
    args = parse_args(__doc__)
    X, _ = datasets.load_bunny()
    rng = np.random.default_rng(0)
    batch = 32

    # 32 randomly rotated, shifted and noisy copies of the bunny.
    angles = rng.uniform(-0.5, 0.5, size=batch)
    Ys = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        Ys.append(X @ R + rng.normal(0, 0.05, 3) + rng.normal(0, 0.002, X.shape))
    Ys = np.stack(Ys)  # (32, 453, 3)

    start = time.perf_counter()
    reg = RigidRegistration(X, Ys, device=args.device, dtype=args.dtype)  # X is shared
    TY, (s, R, t) = reg.register()
    batched = time.perf_counter() - start

    start = time.perf_counter()
    for Y in Ys:
        RigidRegistration(X, Y, device=args.device, dtype=args.dtype).register()
    looped = time.perf_counter() - start

    recovered = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))  # rotation angle about z
    print(f"batched: {batched:.2f} s, loop: {looped:.2f} s ({looped / batched:.1f}x)")
    print(f"converged pairs: {int(reg.converged.sum())}/{batch}")
    print(f"max angle error: {np.abs(recovered + np.degrees(angles)).max():.3f} degrees")


if __name__ == "__main__":
    main()
