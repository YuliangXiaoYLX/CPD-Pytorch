# Migration guide

## From `torchcpd` 0.0.x (this repository before version 1.0)

The name `torchcpd` on PyPI belongs to an unrelated project, so the package was renamed.

```diff
- pip install -e .                     # from a clone
+ pip install cpd-pytorch

- from torchcpd import RigidRegistration
+ from cpd_pytorch import RigidRegistration

- reg = RigidRegistration(**{"X": X, "Y": Y, "device": device})
+ reg = RigidRegistration(X, Y, device=device)     # device is now optional
```

| Before | Now |
|---|---|
| Inputs had to be NumPy arrays | NumPy arrays, tensors or lists |
| `device` was required | Optional: the input tensors' device, else the CPU |
| Results were tensors on `device` | Results have the type of the inputs (NumPy in, NumPy out) |
| `dtype` defaulted to float64 | Inferred from the inputs; `torch.float32` or `torch.float64` |
| `t` had shape `(1, D)` | `t` has shape `(D,)`; `(1, D)` is accepted as an initial value |
| Initial `R`, `B` had to be positive-definite tensors | `R` must be a rotation, `B` any matrix; NumPy works |
| Rigid registration was 2D/3D only | Any dimension |
| `ConstrainedDeformableRegistration` failed in float64 | Works in both precisions |
| Low-rank `S` was a diagonal matrix; `inv_S`, `E` existed | `S` is the vector of eigenvalues |
| Modules `rigidReg`, `affineReg`, `deformReg`, `constDeformReg` | `rigid_registration`, `affine_registration`, `deformable_registration`, `constrained_deformable_registration` |
| Examples read `../data/*.txt` | `cpd_pytorch.datasets.load_fish()` / `load_bunny()` |

Some results differ slightly from version 0.0.1 because of the [fixes](changelog.md): the
affine convergence test uses the correct objective, and registrations of noise-free data
stop once they are exact instead of running to `max_iterations`.

## From pycpd

CPD-PyTorch keeps the pycpd interface: the same classes, parameter names, defaults,
`register(callback)` and `get_registration_parameters()`. In float64 the results agree
with pycpd to about \(10^{-8}\), which the test suite checks for every method. The
differences:

- Results come back as NumPy arrays for NumPy input, but attributes such as `reg.TY`,
  `reg.R` or `reg.sigma2` are tensors. Convert them with `.cpu().numpy()`.
- `X` and `Y` can be passed positionally. In pycpd, `RigidRegistration(X, Y)` silently
  binds `X` to `R`.
- `reg.P` is computed on first access instead of being stored at every iteration.
- The affine objective and the deformable objective (reported as `inf` by pycpd) are
  correct, so `reg.q`, the callback's `error` and the affine stopping iteration can
  differ.
- New options: `device`, `dtype`, `normalize`, `chunk_size`, batches, `correspondences()`.
