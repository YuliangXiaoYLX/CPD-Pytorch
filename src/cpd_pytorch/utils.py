"""Numerical building blocks shared by the registration methods.

All functions accept leading batch dimensions: a point set is a ``(..., n, D)`` tensor.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np
import torch

from ._backend import ArrayLike, as_output, as_tensor, is_numpy_like, resolve_dtype

__all__ = ["gaussian_kernel"]

#: Default upper bound on the number of matrix entries held in one temporary block by the
#: chunked computations (2**26 entries: 256 MiB in float32, 512 MiB in float64).
DEFAULT_MAX_ELEMENTS = 2**26


def squared_distances(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances ``||A_i - B_j||²``.

    Parameters
    ----------
    A, B : torch.Tensor
        Point sets of shape ``(..., m, D)`` and ``(..., n, D)`` with equal batch shapes.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(..., m, n)``.

    Notes
    -----
    The distances are computed from coordinate differences. The faster-looking expansion
    ``|a|² + |b|² - 2 a·b`` cancels catastrophically for nearby points far from the origin
    (it loses every significant digit in float32 on typical scans).

    On the CPU the exact mode of ``torch.cdist`` is the fastest way to do this. On GPUs it
    is a slow path (30-50x slower on CUDA, 4x on MPS), so the squared differences are summed
    one coordinate at a time instead, which needs two ``(..., m, n)`` buffers.
    """
    if A.device.type == "cpu":
        return torch.cdist(A, B, compute_mode="donot_use_mm_for_euclid_dist").square_()
    out = (A[..., :, None, 0] - B[..., None, :, 0]).square_()
    for d in range(1, A.shape[-1]):
        out.add_((A[..., :, None, d] - B[..., None, :, d]).square_())
    return out


def orthonormal_basis(A: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the columns of ``A`` (the Q factor of a reduced QR decomposition)."""
    # torch.linalg.qr hangs on some MPS setups (seen with PyTorch 2.12); the matrix is thin
    # (M x p), so factorizing it on the CPU there is cheap.
    Q: torch.Tensor = torch.linalg.qr(A.cpu() if A.device.type == "mps" else A)[0]
    return Q.to(A.device)


def gaussian_kernel_tensor(A: torch.Tensor, B: torch.Tensor, beta: float) -> torch.Tensor:
    """Gaussian kernel matrix ``exp(-||A_i - B_j||² / (2 β²))`` for tensors."""
    return squared_distances(A, B).div_(-2.0 * beta * beta).exp_()


def gaussian_kernel(X: ArrayLike, beta: float, Y: ArrayLike | None = None) -> ArrayLike:
    r"""Gaussian (RBF) kernel matrix used to regularize deformable registration.

    $$
    G_{ij} = \exp\left(-\frac{\lVert x_i - y_j \rVert^2}{2\beta^2}\right)
    $$

    Parameters
    ----------
    X : array_like, shape (..., m, D)
        First point set.
    beta : float
        Kernel width $\beta > 0$, in the units of the points.
    Y : array_like, shape (..., n, D), optional
        Second point set. Defaults to ``X``.

    Returns
    -------
    numpy.ndarray or torch.Tensor, shape (..., m, n)
        NumPy array if the inputs are NumPy arrays, otherwise a tensor on the device of ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> from cpd_pytorch import gaussian_kernel
    >>> gaussian_kernel(np.zeros((2, 3)), beta=1.0)
    array([[1., 1.],
           [1., 1.]])
    """
    numpy_out = is_numpy_like(X) and (Y is None or is_numpy_like(Y))
    A = as_tensor(X, "X")
    B = A if Y is None else as_tensor(Y, "Y")
    dtype = resolve_dtype(None, A, B)
    K = gaussian_kernel_tensor(A.to(dtype=dtype), B.to(device=A.device, dtype=dtype), float(beta))
    return as_output(K, numpy_out)


def block_rows(rows: int, width: int, batch: int, max_elements: int) -> int:
    """Return how many rows fit in a ``(batch, rows, width)`` block of ``max_elements``."""
    return max(1, min(rows, max_elements // max(1, width * batch)))


def kernel_matmul(
    A: torch.Tensor, B: torch.Tensor, beta: float, V: torch.Tensor, max_elements: int
) -> torch.Tensor:
    """Compute ``gaussian_kernel(A, B, beta) @ V`` without forming the full kernel matrix.

    Rows of the kernel are generated in blocks, so the extra memory is bounded by
    ``max_elements`` entries.
    """
    rows = A.shape[-2]
    batch = math.prod(A.shape[:-2])
    step = block_rows(rows, B.shape[-2], batch, max_elements)
    if step >= rows:
        return gaussian_kernel_tensor(A, B, beta) @ V
    out = V.new_empty(*A.shape[:-1], V.shape[-1])
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        out[..., start:stop, :] = gaussian_kernel_tensor(A[..., start:stop, :], B, beta) @ V
    return out


def _top_eigenpairs(Q: torch.Tensor, S: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the ``k`` eigenpairs of largest magnitude, in descending order."""
    order = S.abs().argsort(dim=-1, descending=True)[..., :k]
    return Q.gather(-1, order.unsqueeze(-2).expand(*Q.shape[:-1], k)), S.gather(-1, order)


def low_rank_eigen(
    Y: torch.Tensor,
    beta: float,
    num_eig: int,
    *,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
    oversampling: int = 20,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading eigenpairs of the Gaussian kernel matrix ``G = gaussian_kernel(Y, Y, beta)``.

    Small problems use a dense symmetric eigendecomposition. Larger ones use randomized
    subspace iteration with Rayleigh-Ritz extraction (Halko, Martinsson & Tropp, 2011),
    which only needs products ``G @ V`` and costs ``O(M² p)`` per iteration instead of
    ``O(M³)``. When ``G`` does not fit in ``max_elements`` entries it is never formed: its
    rows are generated block by block.

    Iteration stops once every requested Ritz pair has a residual
    ``||G q - θ q|| <= tol · θ_k``, where ``θ_k`` is the smallest eigenvalue kept (or the
    rounding level of ``G`` in low precision). The error of ``Q diag(S) Qᵀ`` is then about
    ``tol`` times the truncation error of the rank-``k`` approximation itself, so the
    registration cannot tell the result from a dense eigendecomposition.

    Parameters
    ----------
    Y : torch.Tensor, shape (..., M, D)
        Point set defining the kernel matrix.
    beta : float
        Kernel width.
    num_eig : int
        Number of eigenpairs; clipped to ``M``.
    max_elements : int, optional
        Memory budget (matrix entries) for ``G`` or its row blocks.
    oversampling : int, optional
        Extra subspace dimensions; more converge faster per iteration.
    max_iter : int, optional
        Maximum number of subspace iterations.
    tol : float, optional
        Residual tolerance relative to the smallest kept eigenvalue.
    seed : int, optional
        Seed of the random start, for reproducible results.

    Returns
    -------
    Q : torch.Tensor, shape (..., M, k)
        Orthonormal eigenvectors.
    S : torch.Tensor, shape (..., k)
        Eigenvalues in descending order.
    """
    M = Y.shape[-2]
    k = min(int(num_eig), M)
    batch = math.prod(Y.shape[:-2])
    p = min(k + oversampling, M)
    if 4 * p >= M:  # dense eigendecomposition costs about as much as a few iterations
        S, Q = torch.linalg.eigh(gaussian_kernel_tensor(Y, Y, beta))
        return _top_eigenpairs(Q, S, k)

    G = gaussian_kernel_tensor(Y, Y, beta) if batch * M * M <= max_elements else None
    matmul: Callable[[torch.Tensor], torch.Tensor]
    if G is not None:
        matmul = G.matmul
    else:

        def matmul(V: torch.Tensor) -> torch.Tensor:
            return kernel_matmul(Y, Y, beta, V, max_elements)

    # Residuals cannot drop below the rounding error of the products G @ V.
    rounding = 10 * math.sqrt(M) * torch.finfo(Y.dtype).eps
    generator = torch.Generator().manual_seed(seed)
    V = torch.randn(*Y.shape[:-2], M, p, generator=generator, dtype=Y.dtype).to(Y.device)
    V = orthonormal_basis(V)
    for _ in range(max_iter):
        Z = matmul(V)  # G V
        T = V.transpose(-1, -2) @ Z
        theta, U = torch.linalg.eigh((T + T.transpose(-1, -2)) / 2)
        theta, U = theta.flip(-1), U.flip(-1)  # descending Ritz values
        Q = V @ U
        residual = (Z @ U - Q * theta.unsqueeze(-2)).square().sum(-2).sqrt()  # ||G q - θ q||
        threshold = torch.maximum(
            tol * theta[..., k - 1 : k].abs(), rounding * theta[..., :1].abs()
        )
        if bool((residual[..., :k] <= threshold).all()):
            return Q[..., :k], theta[..., :k]
        V = orthonormal_basis(Z)

    if G is not None:
        S, Q = torch.linalg.eigh(G)
        return _top_eigenpairs(Q, S, k)
    warnings.warn(
        f"The low-rank eigensolver did not reach tol={tol:g} in {max_iter} iterations; the "
        "kernel spectrum decays slowly. Consider a larger beta, a larger num_eig, or "
        "low_rank=False.",
        RuntimeWarning,
        stacklevel=3,
    )
    return Q[..., :k], theta[..., :k]


def is_rotation(R: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check that every ``(..., D, D)`` matrix in ``R`` is a rotation (``RᵀR = I``, ``det = 1``)."""
    eye = torch.eye(R.shape[-1], dtype=R.dtype, device=R.device)
    orthonormal = torch.allclose(R.transpose(-1, -2) @ R, eye.expand_as(R), atol=atol, rtol=0.0)
    return orthonormal and bool((torch.linalg.det(R) > 0).all())


def _as_float(value: ArrayLike) -> float:
    return float(np.asarray(value))
