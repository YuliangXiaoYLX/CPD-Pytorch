# Examples

The [quick-start notebook](https://colab.research.google.com/github/YuliangXiaoYLX/CPD-Pytorch/blob/main/examples/quickstart.ipynb) is the fastest way to try CPD-PyTorch: it runs in
the browser on Google Colab (including a free GPU), with no installation.

The scripts in
[`examples/`](https://github.com/YuliangXiaoYLX/CPD-Pytorch/tree/main/examples) need
matplotlib (`pip install "cpd-pytorch[examples]"`) and run from any directory. Every
script accepts:

| Option | Meaning |
|---|---|
| `--device` | `cpu`, `cuda`, `mps`, ... (default: `cuda` if available, else `cpu`) |
| `--dtype` | `float32` or `float64` (default) |
| `--save FILE` | save an animation (`.gif`) or the final frame (`.png`) |
| `--no-show` | do not open a window |

```bash
python examples/fish_deformable_2D.py --save deformable.gif
```

## Gallery

| Script | Result |
|---|---|
| `bunny_rigid_3D.py`: the Stanford bunny, rotated by 45° and shifted | ![rigid](assets/rigid_bunny.gif){ width="320" } |
| `fish_affine_2D.py`: a rotated, sheared and shifted fish | ![affine](assets/affine_fish.gif){ width="320" } |
| `fish_deformable_2D.py`: two different fish | ![deformable](assets/deformable_fish.gif){ width="320" } |
| `fish_constrained_deformable_2D.py`: a partial fish and four landmarks | ![constrained](assets/constrained_fish.gif){ width="320" } |

## All scripts

| Script | Shows |
|---|---|
| `fish_rigid_2D.py`, `fish_rigid_3D.py`, `bunny_rigid_3D.py` | rigid registration |
| `fish_affine_2D.py`, `fish_affine_3D.py` | affine registration |
| `fish_deformable_2D.py`, `fish_deformable_3D.py` | deformable registration |
| `fish_deformable_3D_lowrank.py` | low-rank deformable registration |
| `fish_constrained_deformable_2D.py`, `fish_constrained_deformable_3D.py` | landmarks |
| `batch_registration.py` | 32 pairs in one call, compared with a loop |
| `large_point_clouds.py` | large point sets, outliers, correspondences (`--points 200000`) |

## Writing your own callback

`register(callback=...)` calls the function after every iteration with the iteration
number, the objective and the current point sets, in the type of the inputs:

```python
def monitor(iteration, error, X, Y):
    print(f"iteration {iteration}: objective {error:.4f}")


reg.register(callback=monitor)
```

The `Animator` class in `examples/_plot.py` is a complete example that draws the
registration live and saves animations.
