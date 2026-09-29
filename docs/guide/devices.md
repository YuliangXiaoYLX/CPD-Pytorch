# Devices, precision and array types

## Array types

Every registration class accepts NumPy arrays, PyTorch tensors or nested lists, and
answers in kind:

| Inputs `X`, `Y` | Returned by `register()`, `transform_point_cloud()`, `correspondences()` |
|---|---|
| NumPy arrays or lists | NumPy arrays |
| tensors | tensors on the computation device |

Attributes such as `reg.TY`, `reg.sigma2` or `reg.R` are always tensors on the computation
device. `transform_point_cloud(Z)` answers in the type of `Z`.

Tensors that require gradients are detached: registration itself is not differentiable.

## Device

```python
RigidRegistration(X, Y, device="cuda")  # also "cpu", "cuda:1", "mps"
```

Without `device`, the computation runs where the input tensors live, or on the CPU for
NumPy input. With NumPy input and `device="cuda"`, the data is copied to the GPU, the
registration runs there and the results come back as NumPy arrays.

GPUs pay off from a few thousand points. On an RTX 3090, rigid registration of 20,000
points takes 49 ms per iteration in float32, against 450 ms on a 10-core CPU in float64.
For very small point sets the CPU can be faster, because every EM iteration waits for the
GPU a few times. See the [benchmarks](../benchmarks.md).

## Precision

`dtype` is `torch.float32` or `torch.float64` (the strings `"float32"`/`"float64"` and the
NumPy types work too). Without it, the dtype follows the inputs: NumPy float64 data runs
in float64 and float32 tensors in float32. Integer inputs use float64 and half precision
is upcast to float32.

- **float64** is the most accurate and the default for NumPy data. It is fast on CPUs but
  slow on most consumer GPUs.
- **float32** is several times faster on GPUs and uses half the memory. Alignment errors
  are typically \(10^{-7}\) to \(10^{-5}\) of the object size (float64 reaches rounding
  level on noise-free data). Internally, point sets are centered before any computation,
  and distances are computed from coordinate differences, not from the \(\lVert a\rVert^2 + \lVert b\rVert^2 - 2a\cdot b\) expansion, which loses all
  precision in float32 when points are far from the origin.

## Apple Silicon (MPS)

```python
RigidRegistration(X, Y, device="mps", dtype=torch.float32)
```

MPS has no float64, so passing `dtype=torch.float64` with `device="mps"` raises an error.
All methods run on MPS. The thin QR factorization of the low-rank solver runs on the CPU
there, because `torch.linalg.qr` can hang on MPS.

## Distance computations

The E-step compares every source point with every target point. On the CPU this uses
the exact mode of `torch.cdist`; on GPUs, where that mode is a slow path, the squared
coordinate differences are accumulated one coordinate at a time. Both are exact; neither
uses the faster but inaccurate \(\lVert a\rVert^2 + \lVert b\rVert^2 - 2a\cdot b\)
expansion.
