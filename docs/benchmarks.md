# Benchmarks

Time per EM iteration for random 3D point sets with `N = M` points. Each number is the
average over 10 iterations, including the setup of the registration (kernel matrix,
low-rank eigenpairs). An untimed warm-up run first absorbs one-time costs such as CUDA
initialization and kernel loading. Memory in parentheses is the peak GPU memory held by
tensors on CUDA, and the peak memory of the whole process on the CPU (about 0.2 GB of which
is PyTorch itself). "Out of memory" means the implementation could not allocate its
buffers.

Reproduce them with [`benchmarks/benchmark.py`](https://github.com/YuliangXiaoYLX/CPD-Pytorch/blob/main/benchmarks/benchmark.py).

## NVIDIA GeForce RTX 3090

24 GB, PyTorch 2.14, CUDA 13.0.

### float32

| Method               |  Points |  CPD-PyTorch 1.0 |   torchcpd 0.0.1 |
| -------------------- | ------: | ---------------: | ---------------: |
| Rigid                |   2,000 | 1.1 ms (0.06 GB) |  1.6 ms (0.1 GB) |
| Rigid                |   8,000 |  8.4 ms (0.5 GB) |   16 ms (1.7 GB) |
| Rigid                |  20,000 |   49 ms (0.8 GB) |  95 ms (10.7 GB) |
| Rigid                |  50,000 |  0.30 s (0.8 GB) |    out of memory |
| Rigid                | 100,000 |   1.2 s (0.8 GB) |    out of memory |
| Affine               |   8,000 |  8.2 ms (0.5 GB) |   17 ms (1.7 GB) |
| Affine               |  50,000 |  0.30 s (0.8 GB) |    out of memory |
| Deformable           |   2,000 | 7.2 ms (0.07 GB) |   10 ms (0.1 GB) |
| Deformable           |   8,000 |   57 ms (0.7 GB) |  153 ms (2.0 GB) |
| Deformable           |  20,000 |  0.42 s (4.6 GB) | 1.37 s (12.2 GB) |
| Deformable, low rank |   2,000 | 1.8 ms (0.06 GB) |  5.1 ms (0.1 GB) |
| Deformable, low rank |   8,000 |   10 ms (0.5 GB) |  126 ms (2.0 GB) |
| Deformable, low rank |  20,000 |   59 ms (0.8 GB) |  1.1 s (12.2 GB) |
| Deformable, low rank |  50,000 |  0.36 s (0.8 GB) |    out of memory |
| Deformable, low rank | 100,000 |   1.4 s (0.8 GB) |    out of memory |

### float64

Consumer GPUs such as the RTX 3090 compute in float64 at a small fraction of their
float32 speed; use float32 on them unless you need the extra precision.

| Method               |  Points | CPD-PyTorch 1.0 |  torchcpd 0.0.1 |
| -------------------- | ------: | --------------: | --------------: |
| Rigid                |   2,000 | 3.4 ms (0.1 GB) | 5.5 ms (0.2 GB) |
| Rigid                |   8,000 |  30 ms (1.0 GB) |  62 ms (3.4 GB) |
| Rigid                |  20,000 | 0.16 s (1.5 GB) |   out of memory |
| Rigid                |  50,000 |  1.0 s (1.6 GB) |   out of memory |
| Rigid                | 100,000 |  4.0 s (1.6 GB) |   out of memory |
| Affine               |   8,000 |  30 ms (1.0 GB) |  72 ms (3.4 GB) |
| Affine               |  50,000 |  1.0 s (1.6 GB) |   out of memory |
| Deformable           |   2,000 |  28 ms (0.1 GB) |  64 ms (0.3 GB) |
| Deformable           |   8,000 | 0.75 s (1.5 GB) |  2.8 s (3.9 GB) |
| Deformable           |  20,000 | 10.4 s (9.2 GB) |   out of memory |
| Deformable, low rank |   2,000 |  15 ms (0.1 GB) |  20 ms (0.3 GB) |
| Deformable, low rank |   8,000 |  92 ms (1.0 GB) | 0.51 s (3.9 GB) |
| Deformable, low rank |  20,000 | 0.72 s (1.6 GB) |   out of memory |
| Deformable, low rank |  50,000 |  4.3 s (1.6 GB) |   out of memory |
| Deformable, low rank | 100,000 | 14.5 s (1.7 GB) |   out of memory |

## Apple M4 Pro CPU, float64

10 performance cores, 24 GB, PyTorch 2.12, NumPy 2.5. torchcpd 0.0.1 and pycpd were not
run beyond 8,000 points: they store `M x N x D` arrays and would need more than 24 GB.

| Method               |  Points | CPD-PyTorch 1.0 |  torchcpd 0.0.1 |             pycpd |
| -------------------- | ------: | --------------: | --------------: | ----------------: |
| Rigid                |   1,000 | 1.3 ms (0.2 GB) | 2.7 ms (0.3 GB) |    15 ms (0.3 GB) |
| Rigid                |   4,000 |  17 ms (0.4 GB) |  52 ms (1.7 GB) |   209 ms (1.7 GB) |
| Rigid                |   8,000 |  69 ms (0.7 GB) | 158 ms (5.4 GB) |   826 ms (5.4 GB) |
| Rigid                |  20,000 | 0.45 s (1.3 GB) |                 |                   |
| Rigid                |  50,000 |  2.9 s (1.4 GB) |                 |                   |
| Rigid                | 100,000 | 11.6 s (1.3 GB) |                 |                   |
| Affine               |   8,000 |  55 ms (0.7 GB) | 132 ms (5.4 GB) |   818 ms (5.4 GB) |
| Deformable           |   1,000 | 4.9 ms (0.2 GB) | 9.7 ms (0.3 GB) |    25 ms (0.3 GB) |
| Deformable           |   2,000 |  29 ms (0.3 GB) |  57 ms (0.6 GB) |    94 ms (0.6 GB) |
| Deformable, low rank |   2,000 |  12 ms (0.3 GB) |  39 ms (0.5 GB) |    85 ms (0.5 GB) |
| Deformable, low rank |   5,000 |  46 ms (0.5 GB) | 734 ms (2.7 GB) | 1,042 ms (2.6 GB) |
| Deformable, low rank |  20,000 |  1.0 s (1.4 GB) |                 |                   |
| Deformable, low rank |  50,000 |  6.7 s (1.7 GB) |                 |                   |
| Deformable, low rank | 100,000 | 22.4 s (1.9 GB) |                 |                   |

## Apple M4 Pro GPU (MPS), float32

20 GPU cores; MPS has no counter for peak memory.

| Method               |  2,000 |  8,000 | 20,000 |        50,000 |       100,000 |
| -------------------- | -----: | -----: | -----: | ------------: | ------------: |
| Rigid                | 3.9 ms |  44 ms | 0.27 s |         1.7 s |         7.1 s |
| Deformable           |  15 ms | 0.25 s |  4.5 s | out of memory | out of memory |
| Deformable, low rank | 5.4 ms |  53 ms | 0.32 s |         2.0 s |         8.0 s |

## Reading the numbers

- The full deformable model needs \(O(M^2)\) memory and \(O(M^3)\) time per iteration.
  Beyond a few thousand points, use `low_rank=True`, which keeps memory linear.
- From about 8,000 points, the RTX 3090 in float32 is 8-17x faster than the M4 Pro CPU in
  float64. The Apple GPU gains less over the fast Apple CPU.

## Run it yourself

```bash
python benchmarks/benchmark.py                                   # CPU, float64, vs pycpd
python benchmarks/benchmark.py --device cuda --dtype float32 --sizes 8000 100000
python benchmarks/benchmark.py --legacy /path/to/old/checkout    # also time torchcpd 0.0.x
```

On a machine with several GPUs, select one with `CUDA_VISIBLE_DEVICES`, e.g.
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python benchmarks/benchmark.py
--device cuda`.
