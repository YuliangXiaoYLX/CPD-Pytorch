"""Affine registration."""

from __future__ import annotations

from typing import Any

import torch

from ._backend import ArrayLike, as_output, is_numpy_like
from .emregistration import EMRegistration

__all__ = ["AffineRegistration"]


class AffineRegistration(EMRegistration):
    r"""Affine registration.

    Estimates a matrix $B$ and a translation $t$ that map the source points
    onto the target points,

    $$
    \mathcal{T}(Y) = Y B + \mathbf{1} t^\top,
    $$

    following Fig. 3 of Myronenko & Song (2010). Points are rows, so ``B`` multiplies from
    the right; it is the transpose of the affine matrix in the paper's notation.

    Parameters
    ----------
    X : array_like, shape (N, D) or (B, N, D)
        Target point set (fixed).
    Y : array_like, shape (M, D) or (B, M, D)
        Source point set (moving).
    B : array_like, shape (D, D) or (B, D, D), optional
        Initial affine matrix. Default: identity.
    t : array_like, shape (D,) or (B, D), optional
        Initial translation. ``(1, D)`` is also accepted. Default: zero.
    **kwargs
        ``sigma2``, ``max_iterations``, ``tolerance``, ``w``, ``normalize``, ``chunk_size``,
        ``device`` and ``dtype``; see [`EMRegistration`][cpd_pytorch.EMRegistration].

    Attributes
    ----------
    B : torch.Tensor, shape (D, D) or (B, D, D)
        Current affine matrix.
    t : torch.Tensor, shape (D,) or (B, D)
        Current translation.

    Examples
    --------
    >>> import numpy as np
    >>> from cpd_pytorch import AffineRegistration
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> Y = X @ np.array([[1.2, 0.3], [-0.1, 0.8]]) + np.array([0.5, -0.2])
    >>> TY, (B, t) = AffineRegistration(X, Y).register()
    >>> bool(np.abs(TY - X).max() < 1e-6)
    True
    """

    _state_attributes = (*EMRegistration._state_attributes, "B", "_t")

    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        B: ArrayLike | None = None,
        t: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(X, Y, **kwargs)
        D = self.D
        if B is None:
            self.B = torch.eye(D, dtype=self.dtype, device=self.device).expand(
                *self.batch_shape, D, D
            )
        else:
            self.B = self._parameter(B, "B", (D, D))
            if not bool(torch.isfinite(self.B).all()):
                raise ValueError("B contains NaN or infinite values.")
        # Translation in the internal frame: t_internal = (t - c_X + c_Y B) / k.
        self._t = (
            self._translation(t) - self._center_X + self._apply(self._center_Y, self.B)
        ) / self._scale.unsqueeze(-1)

    @staticmethod
    def _apply(v: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Row vectors ``v`` (..., D) times ``B`` (..., D, D)."""
        return (v.unsqueeze(-2) @ B).squeeze(-2)

    def update_transform(self) -> None:
        """Closed-form update of ``B`` and ``t`` (Myronenko & Song, 2010, Fig. 3).

        Like the rigid update it only needs the E-step statistics, at
        ``O((M + N) D^2)`` cost.
        """
        X, Y = self._X, self._Y
        Np = self.Np.unsqueeze(-1)
        muX = (self.Pt1.unsqueeze(-1) * X).sum(-2) / Np
        muY = (self.P1.unsqueeze(-1) * Y).sum(-2) / Np
        X_hat = X - muX.unsqueeze(-2)
        Y_hat = Y - muY.unsqueeze(-2)
        self._A = (self.PX - self.P1.unsqueeze(-1) * muX.unsqueeze(-2)).transpose(-1, -2) @ Y_hat
        self._YPY = (Y_hat * self.P1.unsqueeze(-1)).transpose(-1, -2) @ Y_hat
        self._xPx = (self.Pt1 * X_hat.square().sum(-1)).sum(-1)
        self.B = torch.linalg.solve(self._YPY, self._A.transpose(-1, -2))
        self._t = muX - self._apply(muY, self.B)

    def transform_point_cloud(self, Y: ArrayLike | None = None) -> ArrayLike | None:
        """Apply the current transform ``Y @ B + t``.

        Parameters
        ----------
        Y : array_like, shape (K, D) or (B, K, D), optional
            Points to transform. If None, updates `TY` from the source points and
            returns None.

        Returns
        -------
        numpy.ndarray or torch.Tensor or None
            Transformed points, in the array type of ``Y``.
        """
        if Y is None:
            self._TY = self._Y @ self.B + self._t.unsqueeze(-2)
            return None
        B, t = self._external_parameters()
        return as_output(self._points(Y) @ B + t.unsqueeze(-2), is_numpy_like(Y))

    def update_variance(self) -> None:
        """Update the objective ``q``, its change ``diff`` and the variance ``sigma2``."""
        trAB = (self._A.transpose(-1, -2) * self.B).sum((-2, -1))  # tr(A B)
        trBYPYB = (self.B * (self._YPY @ self.B)).sum((-2, -1))  # tr(Bᵀ YPY B)
        previous = self.q
        self.q = (self._xPx - 2 * trAB + trBYPYB) / (
            2 * self._sigma2
        ) + self.D * self.Np / 2 * torch.log(self._sigma2)
        self.diff = (self.q - previous).abs()
        self._set_sigma2((self._xPx - trAB) / (self.Np * self.D))

    @property
    def t(self) -> torch.Tensor:
        """Current translation, shape ``(D,)`` or ``(B, D)``."""
        return self._external_parameters()[1]

    def _external_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Undo the internal frame: t = k t_internal + c_X - c_Y B.
        t = (
            self._scale.unsqueeze(-1) * self._t
            + self._center_X
            - self._apply(self._center_Y, self.B)
        )
        return self.B, t

    def get_registration_parameters(self) -> tuple[Any, Any]:
        """Return the current transform.

        Returns
        -------
        B : array, shape (D, D) or (B, D, D)
            Affine matrix, applied as ``Y @ B``.
        t : array, shape (D,) or (B, D)
            Translation.
        """
        return self._out(self._external_parameters())  # type: ignore[no-any-return]
