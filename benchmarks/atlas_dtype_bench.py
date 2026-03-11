"""Local benchmark for AtlasRegistration float32 vs float64.

Run from repository root:
    PYTHONPATH=. python benchmarks/atlas_dtype_bench.py
"""

import time
import numpy as np

from biocpd import AtlasRegistration


def run_once(dtype, seed, m=2000, d=3, k=30, max_iterations=8, use_kdtree=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, d)).astype(dtype)
    Y = rng.normal(size=(m, d)).astype(dtype)
    U = rng.normal(size=(m * d, k)).astype(dtype)
    L = (np.abs(rng.normal(size=(k,))) + 1e-3).astype(dtype)

    reg = AtlasRegistration(
        X=X,
        Y=Y,
        U=U,
        eigenvalues=L,
        dtype=dtype,
        normalize=True,
        use_kdtree=use_kdtree,
        k=20,
        store_posterior=False,
        max_iterations=max_iterations,
        tolerance=0.0,
    )

    t0 = time.perf_counter()
    reg.register()
    dt = time.perf_counter() - t0
    return dt, reg.iteration, (reg.X.dtype, reg.U_flat.dtype, reg.L.dtype)


def run_many(dtype, repeats=5, **kwargs):
    times = []
    iterations = []
    dtypes = None
    for seed in range(100, 100 + repeats):
        dt, it, internal_dtypes = run_once(dtype, seed, **kwargs)
        times.append(dt)
        iterations.append(it)
        dtypes = internal_dtypes
    return np.array(times), np.array(iterations), dtypes


if __name__ == "__main__":
    t64, i64, d64 = run_many(np.float64)
    t32, i32, d32 = run_many(np.float32)

    print("float64 times:", np.round(t64, 4))
    print("float32 times:", np.round(t32, 4))
    print("float64 mean/std:", float(np.mean(t64)), float(np.std(t64)))
    print("float32 mean/std:", float(np.mean(t32)), float(np.std(t32)))
    print("speedup (64/32):", float(np.mean(t64) / np.mean(t32)))
    print("iters64:", i64)
    print("iters32:", i32)
    print("internal dtypes 64:", d64)
    print("internal dtypes 32:", d32)
