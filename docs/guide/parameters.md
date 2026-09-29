# Choosing parameters

The defaults match pycpd. They work well for clean data of moderate size; the table
lists what to change when they do not.

| Symptom | Try |
|---|---|
| Outliers, noise or only partial overlap | `w=0.1` to `0.3` |
| Deformable result too stiff / too wobbly | smaller / larger `alpha`, or larger / smaller `beta` |
| Parameters that work for one data set fail on another with different units | `normalize=True` |
| Stops too early / runs too long | `max_iterations`, `tolerance` |
| Rigid source shrinks to a point | better initial alignment (`R`, `t`), or `scale=False` |
| Out of memory | `chunk_size`, `low_rank=True`, `dtype=torch.float32` |
| Slow | a GPU for large point sets, `low_rank=True` for deformable |

## Outliers: `w`

`w` is the prior probability that a target point is an outlier (\(0 \le w < 1\)). With
`w=0` every target point must be explained by the source, so outliers, noise and parts of
the target that have no counterpart in the source pull the registration off. Values of
0.1-0.3 are typical for real scans. After registration, `reg.Pt1[n]` is the probability
that target point `n` was explained by the source rather than by the outlier
distribution.

## Units: `normalize`

CPD is not invariant to the units of the data: `beta` and `alpha` (deformable), the
weight of the uniform distribution and the convergence `tolerance` all depend on them.
With `normalize=True`, both point sets are centered on their centroids and divided by
their pooled RMS radius before registration, and every result is mapped back to the
original units. Then

- `beta` is measured in RMS radii of the data (the default 2 gives smooth, global
  deformations),
- the same parameters work for data in millimeters or meters.

A single scale factor is shared by both point sets, so rigid registration without scaling
stays rigid. (The authors' MATLAB implementation scales the two sets separately, which
silently introduces a scale factor.) `normalize=True` is recommended for deformable
registration. It is off by default for compatibility with pycpd.

## Deformable smoothness: `alpha` and `beta`

- `beta`, the kernel width, sets how far the motion of a point influences its
  neighbors. Choose it relative to the size of the features that deform independently.
- `alpha` weighs smoothness against data fit. Increase it if the result overfits noise;
  decrease it if the source cannot follow the target.

Start with the defaults and `normalize=True`, then change one parameter at a time by
factors of 2-10.

## Convergence: `max_iterations` and `tolerance`

Registration stops when the objective (rigid, affine) or \(\sigma^2\) (deformable) changes
by less than `tolerance`, or after `max_iterations`. Check `reg.converged` and
`reg.iteration`, or watch the progress with a callback:

```python
reg.register(callback=lambda iteration, error, X, Y: print(iteration, error))
```

## Initialization

CPD is a local method: it converges to the optimum nearest to the starting pose. It copes
with large translations, but large rotations can end in a wrong optimum. In that case,
start from a coarse alignment (`R=`, `t=`), or run a few initial rotations and keep the
result with the lowest `reg.q`. With scaling enabled and a poor start, the source can
collapse onto the target's centroid; pass `scale=False` or a better start.
