"""Deformable (non-rigid) registration with Gaussian regularization."""

from __future__ import annotations

import math
import numbers
import warnings
from typing import Any

import torch

from ._backend import ArrayLike, as_output, is_numpy_like
from .emregistration import EMRegistration, _positive_int
from .utils import gaussian_kernel_tensor, kernel_matmul, low_rank_eigen

__all__ = ["DeformableRegistration"]

#: ``get_registration_parameters`` does not materialize a low-rank model's kernel matrix
#: when it would hold more entries than this (2**28: 1 GiB in float32, 2 GiB in float64).
_MAX_G_ELEMENTS = 2**28


def _positive(name: str, value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or value <= 0:
        raise ValueError(f"Expected a positive value for {name}, got {value!r}.")
    return float(value)


class DeformableRegistration(EMRegistration):
    r"""Deformable (non-rigid) registration with a Gaussian motion-coherence prior.

    Moves every source point by a smooth displacement field,

    $$
    \mathcal{T}(Y) = Y + G W, \qquad
    G_{ij} = \exp\left(-\frac{\lVert y_i - y_j\rVert^2}{2\beta^2}\right),
    $$

    where $W \in \mathbb{R}^{M \times D}$ minimizes the CPD objective plus the
    regularizer $\frac{\alpha}{2}\operatorname{tr}(W^\top G W)$ (Myronenko & Song,
    2010, Fig. 4). Each M-step solves an ``M x M`` linear system, so the cost grows as
    ``O(M^3)``; use ``low_rank=True`` for large source point sets.

    Parameters
    ----------
    X : array_like, shape (N, D) or (B, N, D)
        Target point set (fixed).
    Y : array_like, shape (M, D) or (B, M, D)
        Source point set (moving).
    alpha : float, optional
        Regularization weight ($\lambda$ in the paper). Larger values give smoother,
        more rigid-like deformations. Default: 2.
    beta : float, optional
        Width of the Gaussian kernel, in data units (in RMS radii with ``normalize=True``).
        Larger values couple the motion of more distant points. Default: 2.
    low_rank : bool, optional
        Approximate $G$ by its ``num_eig`` leading eigenpairs, which makes every
        iteration ``O(M * num_eig^2)`` instead of ``O(M^3)`` and avoids storing ``G`` for
        large ``M`` (Myronenko & Song, 2010, Sec. 4.2). Default: False.
    num_eig : int, optional
        Number of eigenpairs kept by the low-rank approximation. Default: 100.
    **kwargs
        ``sigma2``, ``max_iterations``, ``tolerance``, ``w``, ``normalize``, ``chunk_size``,
        ``device`` and ``dtype``; see [`EMRegistration`][cpd_pytorch.EMRegistration].

    Attributes
    ----------
    W : torch.Tensor, shape (M, D) or (B, M, D)
        Coefficients of the displacement field, in data units.
    G : torch.Tensor, shape (M, M) or (B, M, M)
        Gaussian kernel matrix of the source points (formed on first access in low-rank mode).
    Q, S : torch.Tensor
        Leading eigenvectors ``(..., M, k)`` and eigenvalues ``(..., k)`` of ``G``
        (low-rank mode only).

    Notes
    -----
    Convergence is tested on the change of ``sigma2``, as in pycpd. The objective ``q``
    (reported to the callback as ``error``) includes the regularizer.

    Examples
    --------
    >>> from cpd_pytorch import DeformableRegistration, datasets
    >>> X, Y = datasets.load_fish()
    >>> TY, (G, W) = DeformableRegistration(X, Y).register()
    >>> TY.shape
    (91, 2)
    """

    _state_attributes = (*EMRegistration._state_attributes, "_W")

    low_rank: bool
    Q: torch.Tensor
    S: torch.Tensor
    _W: torch.Tensor
    _GW: torch.Tensor

    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        low_rank: bool = False,
        num_eig: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(X, Y, **kwargs)
        self.alpha = _positive("regularization parameter alpha", alpha, 2.0)
        self.beta = _positive("the width of the Gaussian kernel beta", beta, 2.0)
        self.low_rank = bool(low_rank)
        self.num_eig = _positive_int("num_eig", num_eig)
        self._W = torch.zeros(
            *self.batch_shape, self.M, self.D, dtype=self.dtype, device=self.device
        )
        self._GW = self._W
        self._G: torch.Tensor | None = None
        if self.low_rank:
            self.Q, self.S = low_rank_eigen(
                self._Y, self.beta, self.num_eig, max_elements=self._max_elements
            )
        else:
            self._G = gaussian_kernel_tensor(self._Y, self._Y, self.beta)

    @property
    def G(self) -> torch.Tensor:
        """Gaussian kernel matrix of the source points, shape ``(..., M, M)``."""
        if self._G is None:
            self._G = gaussian_kernel_tensor(self._Y, self._Y, self.beta)
        return self._G

    @property
    def W(self) -> torch.Tensor:
        """Coefficients of the displacement field ``G @ W``, in data units."""
        return self._W * self._scale.unsqueeze(-1).unsqueeze(-1)

    def _system(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagonal weights ``d`` and right-hand side ``F`` of ``(diag(d) G + α σ² I) W = F``."""
        return self.P1, self.PX - self.P1.unsqueeze(-1) * self._Y

    def update_transform(self) -> None:
        """Solve the M-step linear system for ``W`` (Myronenko & Song, 2010, Eq. 22).

        The full version solves ``(diag(P1) G + α σ² I) W = PX - diag(P1) Y`` directly. The
        low-rank version applies the Woodbury identity to ``G ≈ Q diag(S) Qᵀ`` and only
        solves a ``k x k`` system.
        """
        d, F = self._system()
        alpha_sigma2 = self.alpha * self._sigma2
        if not self.low_rank:
            A = d.unsqueeze(-1) * self.G
            A.diagonal(dim1=-2, dim2=-1).add_(alpha_sigma2.unsqueeze(-1))
            self._W = torch.linalg.solve(A, F)
        else:
            Qt = self.Q.transpose(-1, -2)
            dQ = d.unsqueeze(-1) * self.Q
            inner = Qt @ dQ
            inner.diagonal(dim1=-2, dim2=-1).add_(alpha_sigma2.unsqueeze(-1) / self.S)
            self._W = (F - dQ @ torch.linalg.solve(inner, Qt @ F)) / alpha_sigma2[..., None, None]

    def transform_point_cloud(self, Y: ArrayLike | None = None) -> ArrayLike | None:
        """Apply the current displacement field.

        Parameters
        ----------
        Y : array_like, shape (K, D) or (B, K, D), optional
            Points to transform, e.g. a denser version of the source; the displacement is
            interpolated with the Gaussian kernel. If None, updates `TY` from the
            source points and returns None.

        Returns
        -------
        numpy.ndarray or torch.Tensor or None
            Transformed points, in the array type of ``Y``.
        """
        if Y is None:
            if self.low_rank:
                self._GW = self.Q @ (self.S.unsqueeze(-1) * (self.Q.transpose(-1, -2) @ self._W))
            else:
                self._GW = self.G @ self._W
            self._TY = self._Y + self._GW
            return None
        points = self._to_internal(self._points(Y), self._center_Y)
        moved = points + kernel_matmul(points, self._Y, self.beta, self._W, self._max_elements)
        return as_output(self._to_external(moved), is_numpy_like(Y))

    def update_variance(self) -> None:
        """Update ``sigma2`` (Eq. 23), the objective ``q`` and the change ``diff`` of ``sigma2``."""
        sigma2 = self._sigma2
        xPx = (self.Pt1 * self._X.square().sum(-1)).sum(-1)
        yPy = (self.P1 * self._TY.square().sum(-1)).sum(-1)
        trPXY = (self._TY * self.PX).sum((-2, -1))
        residual = xPx - 2 * trPXY + yPy
        regularizer = 0.5 * self.alpha * (self._W * self._GW).sum((-2, -1))  # (α/2) tr(Wᵀ G W)
        self.q = residual / (2 * sigma2) + self.D * self.Np / 2 * torch.log(sigma2) + regularizer
        new_sigma2 = residual / (self.Np * self.D)
        # As in pycpd, the change of sigma2 is the convergence criterion.
        self.diff = (new_sigma2 - sigma2).abs()
        self._set_sigma2(new_sigma2)

    def get_registration_parameters(self) -> tuple[Any, Any]:
        """Return the current transform.

        Returns
        -------
        G : array, shape (M, M) or (B, M, M)
            Gaussian kernel matrix of the source points, such that ``TY = Y + G @ W``
            (plus the offset between the centroids of ``X`` and ``Y`` when
            ``normalize=True``). In low-rank mode ``G`` is None, with a warning, if it
            would hold more than 2**28 entries; use `transform_point_cloud` instead.
        W : array, shape (M, D) or (B, M, D)
            Coefficients of the displacement field.
        """
        G: torch.Tensor | None
        if self._G is None and math.prod(self.batch_shape) * self.M * self.M > _MAX_G_ELEMENTS:
            warnings.warn(
                f"Not forming the {self.M}x{self.M} kernel matrix G for a low-rank model; "
                "returning None. Access reg.G explicitly if you need it, or use "
                "reg.transform_point_cloud() to move points.",
                RuntimeWarning,
                stacklevel=2,
            )
            G = None
        else:
            G = self.G
        return self._out((G, self.W))  # type: ignore[no-any-return]
