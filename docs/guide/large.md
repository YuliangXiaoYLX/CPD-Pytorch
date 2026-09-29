# Large point sets

## Memory of the E-step

The E-step compares every source point with every target point, so a direct
implementation stores an \(M \times N\) matrix, or an \(M \times N \times D\) array of
coordinate differences. At 50,000 points that is 20 GB in float64.

CPD-PyTorch processes the target points in blocks and keeps only what the M-step needs
(\(P\mathbf{1}\), \(P^\top\mathbf{1}\) and \(PX\)), so the memory is \(O(M \times\)
`chunk_size` \()\). By default a block holds at most \(2^{26}\) entries (256 MB in
float32), which is a single block up to about 8,000 x 8,000 points. On a GPU with little
memory, set `chunk_size` (target points per block) explicitly:

```python
reg = RigidRegistration(X, Y, device="cuda", dtype=torch.float32, chunk_size=4096)
```

Blocking changes only the order of summation, not the result.

## The posterior matrix and correspondences

`reg.P` still gives the full posterior matrix of the last E-step, as in pycpd. It is
computed on first access, which needs \(M \times N\) memory.

For large point sets use [`correspondences()`][cpd_pytorch.EMRegistration.correspondences],
which runs block-wise:

```python
index, probability = reg.correspondences(return_probability=True)
# X[n] corresponds to Y[index[n]] with posterior probability probability[n]
```

To flag outliers, use `reg.Pt1[n]`, the probability that target point `n` was explained by
the source at all. With dense point sets the posterior of a target point is shared among
several nearby source points, so `probability` is not a good outlier score.

## Deformable registration

The deformable M-step is the bottleneck: it solves an \(M \times M\) system
(\(O(M^3)\) time, \(O(M^2)\) memory). That is fine up to a few thousand source points on a
CPU and about 20,000 on a GPU. Beyond that, use `low_rank=True`:

```python
reg = DeformableRegistration(
    X, Y, low_rank=True, num_eig=100, normalize=True, device="cuda", dtype=torch.float32
)
```

The low-rank model needs \(O(M \times\) `num_eig` \()\) memory. Its eigenpairs come from
a randomized solver that never forms the \(M \times M\) kernel for large \(M\). The kernel
is also not formed for the parameters returned by `register()`: if it would exceed
\(2^{28}\) entries, `G` is returned as `None` with a warning. Access `reg.G` explicitly if
you need it, or move new points with `reg.transform_point_cloud(points)`.

## Rules of thumb

| Points | Suggestion |
|---|---|
| up to ~5,000 | CPU or GPU; both take milliseconds per iteration |
| 5,000 - 20,000 | a GPU in float32 (an RTX 3090 is about 8x faster than a 10-core CPU) |
| 20,000+ | a GPU in float32, and `low_rank=True` for deformable registration |
| 100,000 | rigid and low-rank deformable take about 1.2-1.4 s per iteration on an RTX 3090 in 0.8 GB; lower `chunk_size` on smaller GPUs |

See the [benchmarks](../benchmarks.md) for measured times and memory.

The script `examples/large_point_clouds.py` registers two resampled bunnies with planted
outliers and reports matches and outliers.
