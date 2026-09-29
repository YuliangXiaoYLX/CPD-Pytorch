# CPD-PyTorch

**Coherent Point Drift (CPD) point set registration in PyTorch: rigid, affine and
deformable, on CPU or GPU, in float32 or float64.**

CPD ([Myronenko & Song, 2010](https://doi.org/10.1109/TPAMI.2010.46)) aligns a moving
point set \(Y\) to a fixed point set \(X\) without known correspondences. The points of
\(Y\) are the centers of a Gaussian mixture, which is fitted to \(X\) by
expectation-maximization while the centers move coherently. CPD-PyTorch implements the
method with the interface of [pycpd](https://github.com/siavashk/pycpd), and adds GPU
support, batching and memory-bounded computation for large point sets.

<table markdown="0">
  <tr>
    <td align="center" width="50%">
      <img src="assets/rigid_bunny.gif" alt="Rigid registration" width="320"><br>
      <b>Rigid</b><br>
      <sub>The bunny, rotated by 45° and shifted: CPD recovers the rotation, translation and scale.</sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/affine_fish.gif" alt="Affine registration" width="320"><br>
      <b>Affine</b><br>
      <sub>A fish outline, rotated, sheared and shifted: CPD recovers the full linear map.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/deformable_fish.gif" alt="Deformable registration" width="320"><br>
      <b>Deformable</b><br>
      <sub>Two different fish: a smooth displacement field bends one onto the other.</sub>
    </td>
    <td align="center" width="50%">
      <img src="assets/constrained_fish.gif" alt="Deformable registration with landmarks" width="320"><br>
      <b>Deformable with landmarks</b><br>
      <sub>A third of the target is missing; four known point pairs (stars) guide the fit.</sub>
    </td>
  </tr>
</table>

## Install

```bash
pip install cpd-pytorch
```

See [Installation](installation.md) for GPU builds and older Python or PyTorch versions.

## Register two point sets

```python
from cpd_pytorch import RigidRegistration, datasets

X, Y = datasets.load_bunny()  # target and source: NumPy arrays (453, 3)
reg = RigidRegistration(X, Y)
TY, (s, R, t) = reg.register()  # TY = s * Y @ R + t is aligned with X
```

The same call works with tensors on a GPU, with batches of point sets and with point
sets of 100,000 points. Every registration class accepts NumPy arrays or tensors and
returns results of the same type.

## Where next

- [Registration methods](guide/methods.md): what each method estimates and how.
- [Choosing parameters](guide/parameters.md): outliers, units, smoothness, convergence.
- [Devices, precision and array types](guide/devices.md): CPU, CUDA, MPS, float32.
- [Batched registration](guide/batching.md) and [large point sets](guide/large.md).
- [Examples](examples.md) and the [API reference](api.md).
- Coming from pycpd or `torchcpd` 0.0.x? Read the [migration guide](migration.md).

## Citation

Please cite both the Coherent Point Drift paper and this library:

```bibtex
@article{myronenko2010cpd,
  title   = {Point Set Registration: Coherent Point Drift},
  author  = {Myronenko, Andriy and Song, Xubo},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {32},
  number  = {12},
  pages   = {2262--2275},
  year    = {2010},
  doi     = {10.1109/TPAMI.2010.46}
}

@software{xiao2026cpdpytorch,
  author  = {Xiao, Yuliang},
  title   = {{CPD-PyTorch}: Coherent Point Drift Point Set Registration in {PyTorch}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/YuliangXiaoYLX/CPD-Pytorch},
  license = {Apache-2.0}
}
```
