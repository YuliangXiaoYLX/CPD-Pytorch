"""Coherent Point Drift (CPD) point set registration in PyTorch.

A PyTorch implementation of Coherent Point Drift (Myronenko & Song, 2010) with the
interface of [pycpd](https://github.com/siavashk/pycpd), running on CPU or GPU in
float32 or float64:

* [`RigidRegistration`][cpd_pytorch.RigidRegistration]: rotation, translation and
  optional isotropic scale.
* [`AffineRegistration`][cpd_pytorch.AffineRegistration]: affine transform.
* [`DeformableRegistration`][cpd_pytorch.DeformableRegistration]: smooth non-rigid
  deformation.
* [`ConstrainedDeformableRegistration`][cpd_pytorch.ConstrainedDeformableRegistration]:
  non-rigid deformation guided by known correspondences.

Every method accepts NumPy arrays or tensors, registers one pair ``(N, D)``/``(M, D)`` or a
batch ``(B, N, D)``/``(B, M, D)``, and returns results in the type of its inputs.

Examples
--------
>>> from cpd_pytorch import RigidRegistration, datasets
>>> X, Y = datasets.load_bunny()
>>> TY, (s, R, t) = RigidRegistration(X, Y).register()
"""

from . import datasets
from ._version import __version__
from .affine_registration import AffineRegistration
from .constrained_deformable_registration import ConstrainedDeformableRegistration
from .deformable_registration import DeformableRegistration
from .emregistration import EMRegistration
from .rigid_registration import RigidRegistration
from .utils import gaussian_kernel

__all__ = [
    "AffineRegistration",
    "ConstrainedDeformableRegistration",
    "DeformableRegistration",
    "EMRegistration",
    "RigidRegistration",
    "__version__",
    "datasets",
    "gaussian_kernel",
]
