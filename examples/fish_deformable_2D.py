"""Deformable (non-rigid) registration of two different 2D fish."""

from _plot import Animator, parse_args
from cpd_pytorch import DeformableRegistration, datasets


def main():
    args = parse_args(__doc__)
    X, Y = datasets.load_fish()

    animate = Animator(args, X, Y, title="deformable")
    # alpha: smoothness (larger = stiffer); beta: kernel width in data units.
    reg = DeformableRegistration(X, Y, alpha=2.0, beta=2.0, device=args.device, dtype=args.dtype)
    TY, (G, W) = reg.register(callback=animate)
    print(f"{reg.iteration} iterations, sigma2 = {float(reg.sigma2):.3e}")
    animate.finish()


if __name__ == "__main__":
    main()
