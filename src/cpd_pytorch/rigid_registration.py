"""Rigid registration, optionally with isotropic scaling."""

from __future__ import annotations

from typing import Any

import torch

from ._backend import ArrayLike, as_output, is_numpy_like
from .emregistration import EMRegistration
from .utils import is_rotation

__all__ = ["RigidRegistration"]


class RigidRegistration(EMRegistration):
    r"""Rigid registration with optional isotropic scaling (a similarity transform).

    Estimates a rotation $R$, a scale $s$ and a translation $t$ that map
    the source points onto the target points,

    $$
    \mathcal{T}(Y) = s\,Y R + \mathbf{1} t^\top,
    $$

    following Fig. 2 of Myronenko & Song (2010). Points are rows, so ``R`` multiplies from
    the right; it is the transpose of the rotation matrix in the paper's column-vector
    notation.

    Parameters
    ----------
    X : array_like, shape (N, D) or (B, N, D)
        Target point set (fixed).
    Y : array_like, shape (M, D) or (B, M, D)
        Source point set (moving).
    R : array_like, shape (D, D) or (B, D, D), optional
        Initial rotation; must be a proper rotation matrix. Default: identity.
    t : array_like, shape (D,) or (B, D), optional
        Initial translation. ``(1, D)`` is also accepted. Default: zero.
    s : float or array_like, optional
        Initial scale, positive. Default: 1.
    scale : bool, optional
        Estimate the scale. If False, ``s`` keeps its initial value and the transform is
        rigid. Default: True.
    **kwargs
        ``sigma2``, ``max_iterations``, ``tolerance``, ``w``, ``normalize``, ``chunk_size``,
        ``device`` and ``dtype``; see [`EMRegistration`][cpd_pytorch.EMRegistration].

    Attributes
    ----------
    R : torch.Tensor, shape (D, D) or (B, D, D)
        Current rotation.
    s : torch.Tensor, shape () or (B,)
        Current scale.
    t : torch.Tensor, shape (D,) or (B, D)
        Current translation.

    Examples
    --------
    >>> import numpy as np
    >>> from cpd_pytorch import RigidRegistration
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 3))
    >>> c, s = np.cos(0.3), np.sin(0.3)
    >>> R_true = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    >>> Y = X @ R_true + np.array([0.5, -0.2, 0.1])
    >>> TY, (scale, R, t) = RigidRegistration(X, Y).register()
    >>> bool(np.abs(TY - X).max() < 1e-6)
    True
    """

    _state_attributes = (*EMRegistration._state_attributes, "R", "s", "_t")

    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        R: ArrayLike | None = None,
        t: ArrayLike | None = None,
        s: float | ArrayLike | None = None,
        scale: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(X, Y, **kwargs)
        D = self.D
        if R is None:
            self.R = torch.eye(D, dtype=self.dtype, device=self.device).expand(
                *self.batch_shape, D, D
            )
        else:
            self.R = self._parameter(R, "R", (D, D))
            if not is_rotation(self.R):
                raise ValueError(
                    f"R must be a {D}x{D} rotation matrix (orthonormal with determinant 1)."
                )
        if s is None:
            self.s = torch.ones(self.batch_shape, dtype=self.dtype, device=self.device)
        else:
            self.s = self._parameter(s, "s", ())
            if not bool((self.s > 0).all()):
                raise ValueError(f"The scale factor s must be positive, got {s!r}.")
        self.scale = bool(scale)
        # Translation in the internal frame: t_internal = (t - c_X + s c_Y R) / k.
        t_external = self._translation(t)
        self._t = (
            t_external - self._center_X + self.s.unsqueeze(-1) * self._apply(self._center_Y, self.R)
        ) / self._scale.unsqueeze(-1)

    @staticmethod
    def _apply(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Row vectors ``v`` (..., D) times ``R`` (..., D, D)."""
        return (v.unsqueeze(-2) @ R).squeeze(-2)

    def update_transform(self) -> None:
        r"""Closed-form update of ``R``, ``s`` and ``t`` (Myronenko & Song, 2010, Fig. 2).

        Uses only the E-step statistics ``P1``, ``Pt1`` and ``PX``: with centered point sets,
        $A = \hat X^\top P^\top \hat Y = (PX - P1\,\mu_x^\top)^\top \hat Y$, so the
        update costs $O((M + N) D^2)$ instead of $O(MND)$.
        """
        X, Y = self._X, self._Y
        Np = self.Np.unsqueeze(-1)
        muX = (self.Pt1.unsqueeze(-1) * X).sum(-2) / Np
        muY = (self.P1.unsqueeze(-1) * Y).sum(-2) / Np
        X_hat = X - muX.unsqueeze(-2)
        Y_hat = Y - muY.unsqueeze(-2)
        self._A = (self.PX - self.P1.unsqueeze(-1) * muX.unsqueeze(-2)).transpose(-1, -2) @ Y_hat
        self._YPY = (self.P1 * Y_hat.square().sum(-1)).sum(-1)
        self._xPx = (self.Pt1 * X_hat.square().sum(-1)).sum(-1)

        # Rotation from the SVD of A with a reflection guard (Lemma 1 and Eq. 9).
        U, _, Vh = torch.linalg.svd(self._A)
        C = torch.ones(*U.shape[:-1], dtype=self.dtype, device=self.device)
        C[..., -1] = torch.linalg.det(U @ Vh)
        self.R = ((U * C.unsqueeze(-2)) @ Vh).transpose(-1, -2)
        self._trAR = (self._A.transpose(-1, -2) * self.R).sum((-2, -1))  # tr(A R)
        if self.scale:
            self.s = self._trAR / self._YPY
        self._t = muX - self.s.unsqueeze(-1) * self._apply(muY, self.R)

    def transform_point_cloud(self, Y: ArrayLike | None = None) -> ArrayLike | None:
        """Apply the current transform ``s * Y @ R + t``.

        Parameters
        ----------
        Y : array_like, shape (K, D) or (B, K, D), optional
            Points to transform, e.g. a denser version of the source. If None, updates
            `TY` from the source points and returns None.

        Returns
        -------
        numpy.ndarray or torch.Tensor or None
            Transformed points, in the array type of ``Y``.
        """
        if Y is None:
            self._TY = self.s[..., None, None] * (self._Y @ self.R) + self._t.unsqueeze(-2)
            return None
        s, R, t = self._external_parameters()
        points = self._points(Y)
        return as_output(s[..., None, None] * (points @ R) + t.unsqueeze(-2), is_numpy_like(Y))

    def update_variance(self) -> None:
        """Update the objective ``q``, its change ``diff`` and the variance ``sigma2``."""
        s = self.s
        previous = self.q
        self.q = (self._xPx - 2 * s * self._trAR + s * s * self._YPY) / (
            2 * self._sigma2
        ) + self.D * self.Np / 2 * torch.log(self._sigma2)
        self.diff = (self.q - previous).abs()
        self._set_sigma2((self._xPx - s * self._trAR) / (self.Np * self.D))

    @property
    def t(self) -> torch.Tensor:
        """Current translation, shape ``(D,)`` or ``(B, D)``."""
        return self._external_parameters()[2]

    def _external_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Undo the internal frame: t = k t_internal + c_X - s c_Y R.
        t = (
            self._scale.unsqueeze(-1) * self._t
            + self._center_X
            - self.s.unsqueeze(-1) * self._apply(self._center_Y, self.R)
        )
        return self.s, self.R, t

    def get_registration_parameters(self) -> tuple[Any, Any, Any]:
        """Return the current transform.

        Returns
        -------
        s : float-like array, shape () or (B,)
            Scale.
        R : array, shape (D, D) or (B, D, D)
            Rotation, applied as ``Y @ R``.
        t : array, shape (D,) or (B, D)
            Translation.
        """
        return self._out(self._external_parameters())  # type: ignore[no-any-return]
