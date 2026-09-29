# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - Unreleased

First release on PyPI. The code was rewritten; see the
[migration guide](https://cpd-pytorch.readthedocs.io/en/latest/migration/) for the details.

### Changed (breaking)

- The package is published as `cpd-pytorch` and imported as `cpd_pytorch`
  ([#3](https://github.com/YuliangXiaoYLX/CPD-Pytorch/issues/3)). The PyPI name `torchcpd` belongs to an unrelated project.
- `X` and `Y` are the first positional arguments; every other argument is keyword-only.
- `device` is optional (the device of the input tensors, otherwise the CPU). `dtype`
  accepts `torch.float32` or `torch.float64` and is otherwise inferred from the inputs;
  float32 halves the memory ([#4](https://github.com/YuliangXiaoYLX/CPD-Pytorch/issues/4)).
- Results are returned in the type of the inputs: NumPy arrays for NumPy input, tensors on
  `device` for tensor input. Attributes such as `reg.TY` are always tensors.
- The rigid and affine translation `t` has shape `(D,)` (was `(1, D)`); `(1, D)` is still
  accepted as an initial value.
- Modules follow PEP 8 names (`rigidReg` → `rigid_registration`, `affineReg` →
  `affine_registration`, `deformReg` → `deformable_registration`, `constDeformReg` →
  `constrained_deformable_registration`).
- In low-rank mode `S` is the vector of eigenvalues (was a diagonal matrix); `inv_S` and
  `E` were removed.

### Added

- Batched registration: pass `(B, N, D)` / `(B, M, D)` arrays to register many pairs in
  one call. Every pair keeps its own parameters and convergence.
- Memory-bounded E-step (`chunk_size`): the `M x N` posterior is never stored, so point
  sets of 100,000 points take less than 1 GB of GPU memory. `reg.P` is computed on demand.
- `correspondences()`: most probable source point (and its probability) for every target
  point.
- `normalize=True`: registration in normalized coordinates, so `w`, `tolerance`, `alpha`
  and `beta` do not depend on the units of the data. A single scale factor is shared by
  both point sets, so rigid registration without scaling stays rigid.
- Faster low-rank deformable registration: a randomized eigensolver replaces the full
  `O(M^3)` eigendecomposition, and never forms the `M x M` kernel for large `M`.
- `converged` attribute, `datasets.load_fish()` / `datasets.load_bunny()`, type hints
  (`py.typed`) and support for float32/float64 on CPU, CUDA and Apple MPS.
- Rigid registration works in any dimension (was limited to 2D and 3D).

### Fixed

- `ConstrainedDeformableRegistration` crashed in float64 with a float/double mismatch.
- The affine objective used `tr(B YPY B)` instead of `tr(Bᵀ YPY B)`. Only the stopping test
  used it, so registrations could stop at the wrong iteration.
- Initial rotations were validated as positive-definite matrices, which rejected valid
  rotations (e.g. 90°). They are now checked for being proper rotations; the affine
  matrix `B` no longer has to be positive definite.
- Initial `R`, `t`, `B` and `s` given as NumPy arrays were rejected, and tensors were not
  moved to the registration's device and dtype.
- The deformable objective reported to the callback was always `inf`; the low-rank
  regularization energy accumulated across iterations.
- When `sigma2` reached zero (noise-free data) it was reset to `tolerance / 10`, so
  iterations continued until `max_iterations`. Registration now stops as converged.
- float32 and float64 treated far-away points differently in the E-step.

### Performance

- The E-step no longer builds `M x N x D` arrays and processes target points in blocks;
  distances are computed exactly with the fastest kernel for each device. Rigid and affine
  updates use `O((M + N) D^2)` statistics instead of `O(M N D)` products, and the
  deformable update no longer multiplies by `M x M` diagonal matrices.
- On an NVIDIA RTX 3090 in float32, compared with 0.0.1: rigid, affine and deformable
  registration are 1.4-3.3x faster, low-rank deformable registration 3-19x faster, with up
  to 15x less GPU memory. Rigid and low-rank registration of 100,000 points take 1.2-1.4 s
  per iteration in 0.8 GB; 0.0.1 runs out of memory from 50,000 points (20,000 in
  float64).
- On a CPU (Apple M4 Pro, float64): rigid registration of 8,000 points is 2.3x faster with
  7x less memory, and low-rank deformable registration of 5,000 points 16x faster.

### Removed

- The `future` dependency. matplotlib is only needed for the examples
  (`pip install "cpd-pytorch[examples]"`).
- Committed build artifacts (`build/`, `dist/`, `*.egg-info`, `__pycache__`).

## [0.0.1] - 2024-03-31

- Initial PyTorch port of pycpd: rigid, affine, deformable and constrained deformable
  registration.

[1.0.0]: https://github.com/YuliangXiaoYLX/CPD-Pytorch/compare/0fd2ec5...HEAD
[0.0.1]: https://github.com/YuliangXiaoYLX/CPD-Pytorch/tree/0fd2ec5
