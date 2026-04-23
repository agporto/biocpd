# AtlasRegistration correctness and optimization review

This review inspects `biocpd/atlas_registration.py` for algorithmic correctness,
numerical stability, and performance hotspots.

## Correctness assessment

### 1) EM flow is internally consistent for atlas mode
- `maximization()` intentionally skips `transform_point_cloud()` because
  `update_transform()` already computes and stores `self.TY` and `self.TY_world`.
- This is consistent with `EMRegistration.iterate()` calling expectation then
  maximization, and avoids redundant work.

Status: **correct**.

### 2) Sparse expectation can silently change precision
- In `_accumulate_sparse_stats()`, denominators are materialized as Python `float`
  (`astype(float)`), so sparse-path probabilities are accumulated in float64 even
  when `dtype=float32` is requested.
- This is numerically safe but creates a dense/sparse precision mismatch and adds
  conversions when writing back into `self.Pt1`, `self.P1`, and `self.PX`.

Status: **correct but inconsistent** with user-selected internal dtype.

### 3) Sparse fallback behavior is safe
- If sparse search yields no valid neighbors (`mask.any() == False`) or no valid
  query rows, the code disables sparse mode and falls back to dense expectation.
- This avoids divide-by-zero and empty-statistics failures.

Status: **correct** and robust.

### 4) Similarity update is well-posed under low support
- `_weighted_similarity_update()` guards the total weight using `tiny` and uses
  weighted centering and SVD-based orthogonal Procrustes.
- Translation update uses `t = mu_x - s * (mu_y @ R.T)`, matching
  `TY = s * (Y @ R.T) + t` convention.

Status: **correct**.

### 5) Variance floor strategy improves stability
- `update_variance()` applies a dataset-scale floor (`_variance_floor`) and
  keeps sigma positive if `Np` is effectively zero.

Status: **correct** for numerical stability.

## Optimization assessment

### Existing optimizations that are good
- Dense expectation is block-wise (`_compute_dense_posterior_stats`) to limit
  peak memory.
- Sparse expectation uses `cKDTree.query(..., workers=-1)` for parallel nearest
  neighbor lookup.
- Cholesky solve (`cho_factor`/`cho_solve`) is used for the coefficient update.

### High-value opportunities

1. **Keep sparse accumulators in `self.dtype`**
   - Replace `astype(float)` with `astype(self.dtype, copy=False)` and ensure
     `norm_weights`/`PX` stay in that dtype.
   - Benefit: lower conversion overhead and consistent float32 behavior.

2. **Avoid repeated `np.repeat(self.P1, self.D)` allocations**
   - `self.P1_ext` is already preallocated; current fill is good, but an
     equivalent reshape/broadcast path may reduce temporary pressure in very
     large `M*D` runs if profiled as hotspot.

3. **Optional: vectorize sparse `PX` accumulation further**
   - Current per-dimension `bincount` loop is efficient and cache-friendly for
     small `D` (2/3), but for higher dimensions an index-add strategy may be
     faster.

4. **Adaptive `k` in sparse mode (optional)**
   - Fixed `k` can be suboptimal as sigma shrinks. Reducing `k` late in
     optimization may cut query and accumulation cost.

## Recommended changes (priority)

1. **(P1)** Dtype consistency in sparse E-step accumulators.
2. **(P2)** Add a regression test asserting sparse-path dtype consistency for
   float32 runs.
3. **(P3)** Consider adaptive `k` as an opt-in performance knob.

## Overall conclusion

`AtlasRegistration` is algorithmically sound and already includes meaningful
stability/performance engineering. The primary actionable improvement is
precision consistency in the sparse path when users request float32.

## Measured float32 vs float64 speed (current branch)

Command:

```bash
PYTHONPATH=. python benchmarks/atlas_dtype_bench.py
```

Observed output on this environment:

- float64 mean/std: `0.2223322654 / 0.0549193897` seconds
- float32 mean/std: `0.1396967496 / 0.0213199717` seconds
- speedup (64/32): `1.5915x` (float32 faster)
- iterations: both dtypes ran exactly 8 iterations for all repeats

Interpretation: with identical benchmark settings in this repository's atlas
benchmark, float32 is currently about **1.6x faster** than float64 while
preserving the configured internal dtype through registration.

## Additional low-hanging speedups

Implemented in code:

1. Removed an extra per-dimension `astype(self.dtype)` copy during sparse `PX`
   accumulation; assignment into preallocated `PX` already casts as needed.
2. Replaced `np.repeat(self.P1, self.D)` with an in-place reshape/broadcast
   write to `self.P1_ext` to avoid creating a temporary `M*D` array each
   iteration.

Still worth trying (small, safe):

- Gate the sparse `csr_matrix` construction behind `store_posterior` only and
  avoid any sparse-matrix allocations when only summary stats are needed.
- Add an adaptive `k` policy in sparse mode (smaller `k` as `sigma2` shrinks).
- Cache `base_rows = np.arange(self.M)` once in `__init__` to avoid rebuilding
  it in sparse expectation.
