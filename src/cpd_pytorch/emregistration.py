"""Expectation-maximization core shared by every CPD registration method.

Coherent Point Drift (Myronenko & Song, 2010) treats the source points ``Y`` as the
centroids of a Gaussian mixture and fits the mixture to the target points ``X`` by
expectation-maximization (EM), moving the centroids coherently through a rigid, affine
or smooth non-rigid transform.
"""

from __future__ import annotations

import math
import numbers
import sys
import warnings
from typing import Any, Callable, Iterator

import numpy as np
import torch

from ._backend import (
    ArrayLike,
    DeviceLike,
    DTypeLike,
    as_output,
    as_tensor,
    check_dtype_device,
    is_numpy_like,
    resolve_device,
    resolve_dtype,
    to_tensor,
)
from .utils import DEFAULT_MAX_ELEMENTS, block_rows, squared_distances

__all__ = ["EMRegistration", "initialize_sigma2"]

Callback = Callable[..., Any]

#: Floor on the per-target normalization of the posterior (the denominator of the E-step).
#: pycpd uses the float64 machine epsilon; the same constant is used for float32 so that
#: both precisions treat far-away target points identically.
_DENOMINATOR_FLOOR = float(np.finfo(np.float64).eps)


def initialize_sigma2(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    r"""Compute the initial variance of the Gaussian mixture.

    $$
    \sigma^2 = \frac{1}{DMN}\sum_{m=1}^{M}\sum_{n=1}^{N}\lVert x_n - y_m\rVert^2
    $$

    Evaluated in $O((M + N) D)$ time and memory from first and second moments,
    after shifting both sets by the mean of ``X`` to avoid cancellation.

    Parameters
    ----------
    X : torch.Tensor, shape (..., N, D)
        Target points.
    Y : torch.Tensor, shape (..., M, D)
        Source points.

    Returns
    -------
    torch.Tensor, shape (...)
        Initial variance for each point-set pair.
    """
    N, D = X.shape[-2:]
    M = Y.shape[-2]
    center = X.mean(-2, keepdim=True)
    Xc, Yc = X - center, Y - center
    total = (
        M * Xc.square().sum((-2, -1))
        + N * Yc.square().sum((-2, -1))
        - 2 * (Xc.sum(-2) * Yc.sum(-2)).sum(-1)
    )
    return total / (D * M * N)


def _posterior_blocks(
    X: torch.Tensor, TY: torch.Tensor, sigma2: torch.Tensor, w: float, max_elements: int
) -> Iterator[tuple[int, int, torch.Tensor]]:
    r"""Yield ``(start, stop, P[..., :, start:stop])``: column blocks of the posterior matrix.

    $$
    P_{mn} = \frac{\exp\left(-\frac{\lVert x_n - T(y_m)\rVert^2}{2\sigma^2}\right)}
                  {\sum_{k=1}^{M}
                   \exp\left(-\frac{\lVert x_n - T(y_k)\rVert^2}{2\sigma^2}\right) + c},
    \qquad c = (2\pi\sigma^2)^{D/2}\,\frac{w}{1-w}\,\frac{M}{N}
    $$
    Each block holds at most ``max_elements`` entries, so memory stays bounded for
    arbitrarily large point sets.
    """
    M, D = TY.shape[-2:]
    N = X.shape[-2]
    batch = math.prod(X.shape[:-2])
    exponent_scale = (-0.5 / sigma2).unsqueeze(-1).unsqueeze(-1)
    c: torch.Tensor | None = None
    if w > 0:
        c = (2 * math.pi * sigma2) ** (D / 2) * (w / (1 - w) * M / N)
        c = c.unsqueeze(-1).unsqueeze(-1)
    step = block_rows(N, M, batch, max_elements)
    for start in range(0, N, step):
        stop = min(start + step, N)
        P = squared_distances(TY, X[..., start:stop, :]).mul_(exponent_scale).exp_()
        denominator = P.sum(-2, keepdim=True).clamp_(min=_DENOMINATOR_FLOOR)
        if c is not None:
            denominator = denominator + c
        yield start, stop, P.div_(denominator)


def _real(name: str, value: object, default: float) -> float:
    """Validate an optional real-valued option."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}.")
    return float(value)


def _iterations(value: object) -> int:
    """Validate ``max_iterations`` (pycpd casts non-integers with a warning)."""
    if value is None:
        return 100
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"max_iterations must be an integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"max_iterations must be non-negative, got {value}.")
    if not isinstance(value, numbers.Integral):
        warnings.warn(
            f"Received a non-integer value for max_iterations: {value}. Casting to integer.",
            stacklevel=4,
        )
    return int(float(value))


def _positive_int(name: str, value: object) -> int:
    """Validate a positive integer option."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _batch_shape(X: torch.Tensor, Y: torch.Tensor) -> torch.Size:
    """Return the common batch shape of two point sets (an unbatched set is shared)."""
    a, b = X.shape[:-2], Y.shape[:-2]
    if a == b or b in ((), (1,)):
        return a
    if a in ((), (1,)):
        return b
    raise ValueError(f"Batch sizes of X {tuple(X.shape)} and Y {tuple(Y.shape)} do not match.")


class EMRegistration:
    r"""Base class implementing the EM loop shared by all CPD registration methods.

    Subclasses implement `update_transform`, `transform_point_cloud`,
    `update_variance` and `get_registration_parameters`.

    Parameters
    ----------
    X : array_like, shape (N, D) or (B, N, D)
        Target point set (fixed). A leading batch dimension registers ``B`` pairs at once.
    Y : array_like, shape (M, D) or (B, M, D)
        Source point set (moving). An unbatched ``X`` or ``Y`` is shared by every pair.
    sigma2 : float or array_like, optional
        Initial variance of the Gaussian mixture, in squared data units. By default it is
        the mean squared distance between all target and source points divided by ``D``.
    max_iterations : int, optional
        Maximum number of EM iterations. Default 100.
    tolerance : float, optional
        Convergence threshold on the change of the objective between iterations
        (on the change of ``sigma2`` for deformable registration). Default ``1e-3``.
    w : float, optional
        Weight of the uniform outlier distribution, ``0 <= w < 1``. Increase it (e.g. to
        0.1-0.3) for noisy data, outliers or partial overlap. Default 0.
    normalize : bool, optional
        Center each point set on its centroid and divide both by their pooled RMS radius
        before registering; results are mapped back to the original units. This makes
        ``w``, ``tolerance`` and the deformable parameters ``alpha``/``beta`` independent
        of the data's units (``beta`` is then measured in RMS radii). Default False,
        which matches pycpd.
    chunk_size : int, optional
        Number of target points processed per block in the E-step. Bounds the E-step
        memory to ``chunk_size * M`` entries instead of ``N * M``. By default blocks of
        at most `DEFAULT_MAX_ELEMENTS` entries are used, which
        is a single block for point sets up to roughly 8000 x 8000 points.
    device : str or torch.device, optional
        Device for the computation, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``.
        Defaults to the device of ``X`` (or ``Y``) if it is a tensor, otherwise the CPU.
    dtype : torch.dtype, optional
        ``torch.float32`` or ``torch.float64``. Defaults to the floating dtype of the
        inputs (float64 for NumPy float64 data, float32 for float32 tensors). float32 is
        much faster on GPUs; float64 is the most accurate.

    Attributes
    ----------
    X, Y : torch.Tensor
        Target and source points on ``device`` with ``dtype``.
    TY : torch.Tensor
        Transformed source points, in the units of ``X``.
    sigma2 : torch.Tensor
        Current variance of the Gaussian mixture, in squared data units.
    iteration : int
        Number of EM iterations performed.
    q : torch.Tensor
        Objective (negative expected complete-data log-likelihood, plus the regularizer
        for deformable registration) evaluated during the last iteration.
    diff : torch.Tensor
        Change of the convergence criterion during the last iteration.
    converged : torch.Tensor
        Boolean tensor, true for pairs that met the convergence criterion before
        ``max_iterations``.
    P : torch.Tensor
        Posterior matrix of the last E-step, shape ``(..., M, N)``: ``P[m, n]`` is the
        probability that target point ``n`` was generated by source point ``m``.
        Computed on first access, which needs ``M * N`` memory.
    P1, Pt1, PX, Np : torch.Tensor
        Sufficient statistics of the last E-step: ``P @ 1``, ``Pᵀ @ 1``, ``P @ X`` (in the
        internal centered/normalized frame) and ``sum(P)``.
    N, M, D : int
        Number of target points, number of source points and dimension.
    batch_shape : torch.Size
        ``()`` for a single registration, ``(B,)`` for a batch.
    """

    #: Per-pair state tensors, frozen in batched mode once a pair has converged.
    _state_attributes: tuple[str, ...] = (
        "_TY",
        "_sigma2",
        "q",
        "diff",
        "P1",
        "Pt1",
        "PX",
        "Np",
        "_TY_estep",
        "_sigma2_estep",
    )

    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        sigma2: float | ArrayLike | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        w: float | None = None,
        normalize: bool = False,
        chunk_size: int | None = None,
        device: DeviceLike | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        self.max_iterations = _iterations(max_iterations)
        self.tolerance = _real("tolerance", tolerance, 1e-3)
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}.")
        self.w = _real("w", w, 0.0)
        if not 0 <= self.w < 1:
            raise ValueError(f"w must be in [0, 1), got {self.w}.")
        self.chunk_size = None if chunk_size is None else _positive_int("chunk_size", chunk_size)
        self.normalize = bool(normalize)

        X_t, Y_t = as_tensor(X, "X"), as_tensor(Y, "Y")
        self.device = resolve_device(device, X, Y)
        self.dtype = resolve_dtype(dtype, X_t, Y_t)
        check_dtype_device(self.dtype, self.device)
        self._numpy_io = is_numpy_like(X) and is_numpy_like(Y)
        X_t = X_t.to(device=self.device, dtype=self.dtype)
        Y_t = Y_t.to(device=self.device, dtype=self.dtype)
        for name, points in (("X", X_t), ("Y", Y_t)):
            if points.dim() not in (2, 3) or points.shape[-2] == 0 or points.shape[-1] == 0:
                raise ValueError(
                    f"{name} must have shape (n, D) or (B, n, D) with n, D > 0, "
                    f"got {tuple(points.shape)}."
                )
            if not bool(torch.isfinite(points).all()):
                raise ValueError(f"{name} contains NaN or infinite values.")
        if X_t.shape[-1] != Y_t.shape[-1]:
            raise ValueError(
                "X and Y must have the same dimension, "
                f"got D={X_t.shape[-1]} and D={Y_t.shape[-1]}."
            )
        self.batch_shape = _batch_shape(X_t, Y_t)
        self.X = X_t.expand(*self.batch_shape, *X_t.shape[-2:])  # (..., N, D)
        self.Y = Y_t.expand(*self.batch_shape, *Y_t.shape[-2:])  # (..., M, D)
        self.N, self.D = self.X.shape[-2:]
        self.M = self.Y.shape[-2]
        batch = math.prod(self.batch_shape)
        self._max_elements = (
            DEFAULT_MAX_ELEMENTS if self.chunk_size is None else self.chunk_size * self.M * batch
        )

        # Internal frame: registration runs on centered (and optionally rescaled) copies.
        # Shifting both sets by the same vector changes nothing mathematically but avoids
        # cancellation in float32 when the data sit far from the origin.
        self._center_X = self.X.mean(-2)
        if self.normalize:
            self._center_Y = self.Y.mean(-2)
            spread = (self.X - self._center_X.unsqueeze(-2)).square().sum((-2, -1)) + (
                self.Y - self._center_Y.unsqueeze(-2)
            ).square().sum((-2, -1))
            scale = (spread / (self.N + self.M)).sqrt()
            self._scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        else:
            self._center_Y = self._center_X
            self._scale = torch.ones(self.batch_shape, dtype=self.dtype, device=self.device)
        self._X = self._to_internal(self.X, self._center_X)
        self._Y = self._to_internal(self.Y, self._center_Y)
        self._TY = self._Y

        if sigma2 is None:
            self._sigma2 = initialize_sigma2(self._X, self._Y)
        else:
            sigma2_t = self._parameter(sigma2, "sigma2", ())
            if not bool((sigma2_t > 0).all()):
                raise ValueError(f"sigma2 must be positive, got {sigma2!r}.")
            self._sigma2 = sigma2_t / self._scale.square()
        # Registration has converged to machine precision once sigma2 reaches this floor.
        self._sigma2_floor = torch.finfo(self.dtype).eps * self._sigma2

        dtype_, device_ = self.dtype, self.device
        self.iteration = 0
        self.q = torch.full(self.batch_shape, math.inf, dtype=dtype_, device=device_)
        self.diff = torch.full(self.batch_shape, math.inf, dtype=dtype_, device=device_)
        self.P1 = torch.zeros((*self.batch_shape, self.M), dtype=dtype_, device=device_)
        self.Pt1 = torch.zeros((*self.batch_shape, self.N), dtype=dtype_, device=device_)
        self.PX = torch.zeros((*self.batch_shape, self.M, self.D), dtype=dtype_, device=device_)
        self.Np = torch.zeros(self.batch_shape, dtype=dtype_, device=device_)
        self._TY_estep: torch.Tensor | None = None
        self._sigma2_estep: torch.Tensor | None = None
        self._P: torch.Tensor | None = None
        self._active = torch.ones(self.batch_shape, dtype=torch.bool, device=self.device)
        self._floored = torch.zeros(self.batch_shape, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ public API

    def register(self, callback: Callback | None = None) -> tuple[ArrayLike, tuple[Any, ...]]:
        """Run EM until convergence or ``max_iterations``.

        Parameters
        ----------
        callback : callable, optional
            Called after every iteration as ``callback(iteration=..., error=..., X=..., Y=...)``
            where ``error`` is the objective ``q`` (a float, or an array for a batch), ``X``
            the target and ``Y`` the current transformed source, in the type of the inputs.
            Use it to monitor or animate the registration.

        Returns
        -------
        TY : numpy.ndarray or torch.Tensor
            Transformed source points, shape ``(M, D)`` or ``(B, M, D)``.
        parameters : tuple
            Transform parameters; see `get_registration_parameters` of each method.

        Notes
        -----
        Outputs are NumPy arrays when ``X`` and ``Y`` are NumPy arrays (or lists), and
        tensors on ``device`` otherwise.
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and bool(self._active.any()):
            self.iterate()
            if callable(callback):
                error: Any = float(self.q) if not self.batch_shape else self._out(self.q)
                callback(
                    iteration=self.iteration, error=error, X=self._out(self.X), Y=self._out(self.TY)
                )
        return self._out(self.TY), self.get_registration_parameters()

    def iterate(self) -> None:
        """Perform one EM iteration (E-step, then M-step)."""
        previous = self._state() if self.batch_shape else None
        self.expectation()
        self.maximization()
        self.iteration += 1
        if previous is not None:
            self._freeze_converged(previous)
        self._active = self._active & (self.diff > self.tolerance) & ~self._floored

    def expectation(self) -> None:
        """E-step: posterior-weighted statistics ``P1``, ``Pt1``, ``PX`` and ``Np``.

        The posterior matrix ``P`` itself is not stored; it is processed in blocks of at
        most ``chunk_size`` target points, so memory grows as ``O(M * chunk_size)``.
        """
        P1 = torch.zeros_like(self.P1)
        Pt1 = torch.empty_like(self.Pt1)
        PX = torch.zeros_like(self.PX)
        for start, stop, P in _posterior_blocks(
            self._X, self._TY, self._sigma2, self.w, self._max_elements
        ):
            Pt1[..., start:stop] = P.sum(-2)
            P1 += P.sum(-1)
            PX += P @ self._X[..., start:stop, :]
        self.P1, self.Pt1, self.PX, self.Np = P1, Pt1, PX, P1.sum(-1)
        self._TY_estep, self._sigma2_estep, self._P = self._TY, self._sigma2, None

    def maximization(self) -> None:
        """M-step: update the transform, apply it, then update the variance."""
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()

    def correspondences(
        self, return_probability: bool = False
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """Most probable source point for every target point, at the current alignment.

        Parameters
        ----------
        return_probability : bool, optional
            Also return the posterior probability of each match.

        Returns
        -------
        index : numpy.ndarray or torch.Tensor of int64, shape (N,) or (B, N)
            ``X[n]`` corresponds to ``Y[index[n]]``.
        probability : numpy.ndarray or torch.Tensor, shape (N,) or (B, N)
            Returned if ``return_probability``. Low values flag target points that are
            poorly explained by any source point (outliers or unmatched regions).

        Notes
        -----
        Computed block-wise, so it works for point sets too large for `P`.
        """
        index = torch.empty(*self.batch_shape, self.N, dtype=torch.long, device=self.device)
        probability = torch.empty(*self.batch_shape, self.N, dtype=self.dtype, device=self.device)
        for start, stop, P in _posterior_blocks(
            self._X, self._TY, self._sigma2, self.w, self._max_elements
        ):
            best = P.max(dim=-2)
            probability[..., start:stop] = best.values
            index[..., start:stop] = best.indices
        if return_probability:
            return self._out(index), self._out(probability)
        return self._out(index)

    @property
    def TY(self) -> torch.Tensor:
        """Transformed source points, in the units of ``X``."""
        return self._to_external(self._TY)

    @property
    def sigma2(self) -> torch.Tensor:
        """Current variance of the Gaussian mixture, in squared data units."""
        return self._sigma2 * self._scale.square()

    @property
    def converged(self) -> torch.Tensor:
        """Whether each pair met the convergence criterion before ``max_iterations``."""
        return ~self._active

    @property
    def P(self) -> torch.Tensor:
        """Dense posterior matrix of the last E-step, shape ``(..., M, N)``."""
        if self._P is None:
            if self._TY_estep is None or self._sigma2_estep is None:
                return torch.zeros(
                    *self.batch_shape, self.M, self.N, dtype=self.dtype, device=self.device
                )
            ((_, _, self._P),) = _posterior_blocks(
                self._X, self._TY_estep, self._sigma2_estep, self.w, sys.maxsize
            )
        return self._P

    def __repr__(self) -> str:
        batch = f", batch_shape={tuple(self.batch_shape)}" if self.batch_shape else ""
        return (
            f"{type(self).__name__}(N={self.N}, M={self.M}, D={self.D}{batch}, "
            f"device={self.device}, dtype={self.dtype}, iteration={self.iteration})"
        )

    # ----------------------------------------------------------- subclass interface

    def update_transform(self) -> None:
        """Update the transform parameters (M-step). Implemented by subclasses."""
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes."
        )

    def transform_point_cloud(self, Y: ArrayLike | None = None) -> ArrayLike | None:
        """Apply the current transform. Implemented by subclasses."""
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes."
        )

    def update_variance(self) -> None:
        """Update ``sigma2`` and the objective (M-step). Implemented by subclasses."""
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child "
            "classes."
        )

    def get_registration_parameters(self) -> tuple[Any, ...]:
        """Return the transform parameters. Implemented by subclasses."""
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    # --------------------------------------------------------------------- helpers

    def _out(self, value: Any) -> Any:
        """Return ``value`` as NumPy if the inputs were NumPy, else as tensors."""
        return as_output(value, self._numpy_io)

    def _to_internal(self, points: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return (points - center.unsqueeze(-2)) / self._scale.unsqueeze(-1).unsqueeze(-1)

    def _to_external(self, points: torch.Tensor) -> torch.Tensor:
        """Map points from the internal frame to the frame of ``X``."""
        return points * self._scale.unsqueeze(-1).unsqueeze(-1) + self._center_X.unsqueeze(-2)

    def _points(self, Y: ArrayLike) -> torch.Tensor:
        """Validate and convert user points to be transformed."""
        points = to_tensor(Y, "Y", self.dtype, self.device)
        if points.dim() not in (2, 3) or points.shape[-1] != self.D:
            raise ValueError(
                f"Y must have shape (n, {self.D}) or (B, n, {self.D}), got {tuple(points.shape)}."
            )
        return points

    def _set_sigma2(self, sigma2: torch.Tensor) -> None:
        """Store the new variance, flooring it once the fit is exact to machine precision."""
        self._floored = sigma2 <= self._sigma2_floor
        self._sigma2 = torch.where(self._floored, self._sigma2_floor, sigma2)

    def _state(self) -> dict[str, torch.Tensor | None]:
        return {name: getattr(self, name) for name in self._state_attributes}

    def _freeze_converged(self, previous: dict[str, torch.Tensor | None]) -> None:
        """Restore the state of pairs that had already converged before this iteration."""
        done = ~self._active
        if not bool(done.any()):
            return
        for name, old in previous.items():
            new = getattr(self, name)
            if old is None or new is None:
                continue
            mask = done.reshape(done.shape + (1,) * (new.dim() - done.dim()))
            setattr(self, name, torch.where(mask, old, new))

    def _translation(self, t: ArrayLike | None) -> torch.Tensor:
        """Convert an initial translation to shape ``batch_shape + (D,)`` (``(1, D)`` works too)."""
        if t is None:
            return torch.zeros(*self.batch_shape, self.D, dtype=self.dtype, device=self.device)
        tensor = to_tensor(t, "t", self.dtype, self.device)
        if tensor.dim() >= 2 and tensor.shape[-2] == 1:
            tensor = tensor.squeeze(-2)
        return self._parameter(tensor, "t", (self.D,))

    def _parameter(self, value: ArrayLike, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        """Convert an initial transform parameter and broadcast it to ``batch_shape + shape``."""
        tensor = to_tensor(value, name, self.dtype, self.device)
        try:
            return tensor.expand(torch.Size((*self.batch_shape, *shape)))
        except RuntimeError:
            raise ValueError(
                f"{name} must have shape {shape} or {tuple(self.batch_shape) + shape}, "
                f"got {tuple(tensor.shape)}."
            ) from None
