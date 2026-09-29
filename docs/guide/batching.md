# Batched registration

A leading batch dimension registers \(B\) point-set pairs in one call, with vectorized
tensor operations instead of a Python loop:

```python
import numpy as np
from cpd_pytorch import RigidRegistration

Xs = np.stack(targets)  # (B, N, D)
Ys = np.stack(sources)  # (B, M, D)
reg = RigidRegistration(Xs, Ys)
TY, (s, R, t) = reg.register()
# TY: (B, M, D), s: (B,), R: (B, D, D), t: (B, D)
```

- All pairs in a batch must have the same `N`, `M` and `D`. Register point sets of
  different sizes separately.
- An unbatched `X` of shape `(N, D)` (or `Y` of shape `(M, D)`) is shared by every pair:
  e.g. one template registered to many scans, without copying it.
- Every pair has its own \(\sigma^2\), transform and convergence test. A pair that has
  converged is frozen while the others continue, so the results equal those of
  registering each pair on its own. `reg.converged` has shape `(B,)`; `reg.iteration`
  counts the iterations of the slowest pair.
- Initial parameters can be shared (`R` of shape `(D, D)`) or per pair (`(B, D, D)`).
- For [`ConstrainedDeformableRegistration`][cpd_pytorch.ConstrainedDeformableRegistration]
  the same correspondences apply to every pair.
- The callback receives `error` as an array of shape `(B,)`.

Batching is most useful on a GPU, where many small registrations would otherwise leave it
idle. The example `examples/batch_registration.py` registers 32 bunnies and compares the
time with a loop.
