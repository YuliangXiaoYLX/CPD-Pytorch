"""Command-line options and live plotting shared by the examples (requires matplotlib)."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation


def parse_args(description: str | None, add_arguments=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu, cuda, cuda:1, mps, ... (default: cuda if available, else cpu)",
    )
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    parser.add_argument(
        "--save", metavar="FILE", help="save an animation (.gif) or the last frame (.png)"
    )
    if add_arguments is not None:
        add_arguments(parser)
    return parser.parse_args()


def _limits(point_sets, margin=0.08):
    """Equal-aspect bounds (a square or cube) that contain every point set."""
    points = np.vstack(point_sets)
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    half = (high - low).max() / 2 * (1 + margin)
    return center - half, center + half


class Animator:
    """Registration callback that draws the target (red) and the moving source (blue).

    Create it before registering, pass it to ``register(callback=...)``, then call
    :meth:`finish`. All frames share the same axis limits and every figure has the same
    size, so animations do not jump and line up side by side.
    """

    def __init__(self, args, X, Y, title="", highlight=None, view=None):
        self.show = not args.no_show
        self.save = args.save
        self.title = title
        self.highlight = highlight  # optional indices of landmark points to emphasize
        self.view = view  # optional (elevation, azimuth) of a 3D plot
        self.frames = [("start", None, np.asarray(X), np.asarray(Y))]
        self.limits = _limits([np.asarray(X), np.asarray(Y)])
        self.figure = None

    def __call__(self, iteration, error, X, Y):
        self.frames.append((f"iteration {iteration}", error, np.asarray(X), np.asarray(Y)))
        if self.show:
            self._draw(*self.frames[-1])
            plt.pause(0.001)

    def _draw(self, label, error, X, Y):
        three_d = X.shape[1] == 3
        if self.figure is None:
            self.figure = plt.figure(figsize=(6, 5))
            self.figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
            self.axes = self.figure.add_subplot(111, projection="3d" if three_d else None)
        ax = self.axes
        ax.cla()
        ax.scatter(*X.T, color="tab:red", s=8, label="target X")
        ax.scatter(*Y.T, color="tab:blue", s=8, label="source Y")
        if self.highlight is not None:
            ax.scatter(
                *Y[self.highlight].T, color="tab:green", marker="*", s=150, label="landmarks"
            )
        low, high = self.limits
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        if three_d:
            ax.set_zlim(low[2], high[2])
            try:  # fill the frame like the 2D plots (zoom needs matplotlib >= 3.7)
                ax.set_box_aspect((1, 1, 1), zoom=1.35)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            if self.view is not None:
                ax.view_init(*self.view)
        else:
            ax.set_aspect("equal")
        ax.set_axis_off()
        suffix = "" if error is None else f" · q = {float(error):.4g}"
        ax.set_title(f"{self.title} · {label}{suffix}")
        ax.legend(loc="upper left")

    def finish(self):
        """Save the animation or last frame if requested, then keep the window open."""
        if self.save:
            if self.save.endswith(".gif"):
                step = max(1, (len(self.frames) - 1) // 60)  # at most ~60 frames
                frames = (
                    [self.frames[0]] * 5  # show the start for 0.5 s
                    + self.frames[1::step]
                    + [self.frames[-1]] * 10  # hold the result for 1 s
                )
                # One set of limits for the whole animation, so nothing leaves the frame.
                self.limits = _limits([p for _, _, X, Y in frames for p in (X, Y)])
                self._draw(*frames[0])
                movie = animation.FuncAnimation(
                    self.figure, lambda i: self._draw(*frames[i]), frames=len(frames)
                )
                movie.save(self.save, writer=animation.PillowWriter(fps=10), dpi=80)
            else:
                self._draw(*self.frames[-1])
                self.figure.savefig(self.save, dpi=150)
            print(f"Saved {self.save}")
        if self.show:
            plt.show()
