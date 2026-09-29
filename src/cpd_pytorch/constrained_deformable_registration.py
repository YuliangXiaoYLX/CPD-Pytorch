"""Deformable registration with known point correspondences."""

from __future__ import annotations

from typing import Any

import torch

from ._backend import ArrayLike, as_tensor
from .deformable_registration import DeformableRegistration, _positive

__all__ = ["ConstrainedDeformableRegistration"]


class ConstrainedDeformableRegistration(DeformableRegistration):
    r"""Deformable registration guided by known correspondences (landmarks).

    Extends [`DeformableRegistration`][cpd_pytorch.DeformableRegistration] with
    correspondence priors as in the Extended CPD of Golyanik et al. (2016): source point
    ``source_id[i]`` is encouraged to land on target point ``target_id[i]``. The M-step
    system becomes

    $$
    \left(\operatorname{d}(P\mathbf{1} + \tfrac{\sigma^2}{\alpha_e}\tilde P\mathbf{1})\,G
    + \lambda\sigma^2 I\right) W
    = PX - \operatorname{d}(P\mathbf{1})Y + \tfrac{\sigma^2}{\alpha_e}
      \left(\tilde P X - \operatorname{d}(\tilde P\mathbf{1}) Y\right),
    $$
    where $\tilde P$ is the binary correspondence matrix.

    Parameters
    ----------
    X : array_like, shape (N, D) or (B, N, D)
        Target point set (fixed).
    Y : array_like, shape (M, D) or (B, M, D)
        Source point set (moving).
    e_alpha : float, optional
        Reliability of the correspondences, from ``1e-8`` (very reliable) to ``1``
        (unreliable). Default: ``1e-8``.
    source_id, target_id : array_like of int, shape (K,)
        Indices of corresponding source and target points (required). In batched mode the
        same correspondences are used for every pair.
    **kwargs
        ``alpha``, ``beta``, ``low_rank``, ``num_eig`` (see
        [`DeformableRegistration`][cpd_pytorch.DeformableRegistration]) and ``sigma2``,
        ``max_iterations``, ``tolerance``, ``w``, ``normalize``, ``chunk_size``,
        ``device``, ``dtype`` (see [`EMRegistration`][cpd_pytorch.EMRegistration]).

    Examples
    --------
    >>> import numpy as np
    >>> from cpd_pytorch import ConstrainedDeformableRegistration, datasets
    >>> X, Y = datasets.load_fish()
    >>> ids = np.array([1, 10, 20, 30])
    >>> reg = ConstrainedDeformableRegistration(X, Y, source_id=ids, target_id=ids)
    >>> TY, (G, W) = reg.register()
    """

    def __init__(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        e_alpha: float | None = None,
        source_id: ArrayLike | None = None,
        target_id: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(X, Y, **kwargs)
        self.e_alpha = _positive("regularization parameter e_alpha", e_alpha, 1e-8)
        if source_id is None or target_id is None:
            raise ValueError("source_id and target_id are required (1-D arrays of point indices).")
        self.source_id = self._indices(source_id, "source_id", self.M)
        self.target_id = self._indices(target_id, "target_id", self.N)
        if self.source_id.shape != self.target_id.shape:
            raise ValueError(
                f"source_id and target_id must have the same length, got "
                f"{self.source_id.numel()} and {self.target_id.numel()}."
            )
        # Binary correspondence matrix P~ (duplicate pairs count once), stored as its
        # sufficient statistics P~1 and P~X instead of a dense M x N matrix.
        pairs = torch.unique(torch.stack([self.source_id, self.target_id], dim=1), dim=0)
        source, target = pairs[:, 0], pairs[:, 1]
        self.P1_tilde = torch.zeros(self.M, dtype=self.dtype, device=self.device).index_add_(
            0, source, torch.ones(len(source), dtype=self.dtype, device=self.device)
        )
        self.PX_tilde = torch.zeros_like(self._Y).index_add_(-2, source, self._X[..., target, :])

    def _indices(self, value: ArrayLike, name: str, size: int) -> torch.Tensor:
        index = as_tensor(value, name)
        if (
            index.dim() != 1
            or index.is_floating_point()
            or index.is_complex()
            or index.dtype == torch.bool
        ):
            raise ValueError(f"The {name} must be a 1D array of integer indices.")
        index = index.to(device=self.device, dtype=torch.long)
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= size):
            raise ValueError(f"The {name} must contain indices in [0, {size}).")
        return index

    def _system(self) -> tuple[torch.Tensor, torch.Tensor]:
        d, F = super()._system()
        prior = (self._sigma2 / self.e_alpha).unsqueeze(-1)
        d = d + prior * self.P1_tilde
        F = F + prior.unsqueeze(-1) * (self.PX_tilde - self.P1_tilde.unsqueeze(-1) * self._Y)
        return d, F
