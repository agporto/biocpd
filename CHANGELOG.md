# Changelog

All notable changes in this fork are documented in this file.

## 0.2.0 - 2026-03-12

Changes in this release summarize the full diff between `main` and the current branch.

### Added

- Added `AtlasRegistration`, an SSM/PCA-based CPD variant with optional normalization, weighted similarity optimization, and sparse k-NN expectation updates.
- Added dense blocked posterior-stat accumulation in the EM base to reduce memory pressure for large dense expectation steps.
- Added `store_posterior` control for atlas and dense posterior-stat paths so large registrations can skip retaining the full posterior matrix.
- Added dtype-controlled execution paths for atlas and deformable registration, including explicit `float32` support.
- Added atlas benchmark scripts in `benchmarks/atlas_dtype_bench.py` and `benchmarks/atlas_registration_bench.py`.
- Added a comprehensive regression test suite in `tests/test_regressions.py` covering registration robustness, dtype handling, dense blocked stats, and float32/float64 parity checks.
- Added CI test execution with `pytest` across the existing GitHub Actions matrix.

### Changed

- Bumped the package metadata and modernized packaging around `pyproject.toml` and a minimal `setup.py` shim.
- Updated README examples and option descriptions for atlas registration, normalized `mean_shape`, and dtype usage.
- Deformable, constrained deformable, and atlas registration now default to `np.float32`; callers can still request `np.float64` explicitly.
- The deformable dense expectation step now computes posterior statistics without storing the dense `P` matrix.
- Atlas dense expectation now reuses the blocked EM base implementation instead of materializing a full dense posterior matrix by default.
- Atlas sparse expectation now accumulates posterior statistics directly and supports optional posterior retention only when requested.
- Atlas internal work arrays now consistently follow the configured dtype instead of silently drifting to `float64`.
- Rigid and affine sparse k-NN expectation steps now clamp `k` to the available target size and guard invalid neighbor indices.
- EM initialization now casts integer inputs to floating point internally and uses a mean-centered `sigma2` initialization that matches the full pairwise reference value.

### Fixed

- Fixed atlas validation for missing `U`, negative `lambda_reg`, invalid `kdtree_radius_scale`, and invalid normalized `mean_shape` shapes.
- Fixed atlas transformed-point handling to reuse cached deformations and avoid repeated recomputation of the basis projection.
- Fixed sparse rigid and affine expectation paths for `k > N` and invalid neighbor outputs from `cKDTree.query`.
- Fixed deformable and constrained deformable internal allocations so `float32` execution stays in `float32` through `W`, transformed points, and constraint priors.

### Performance

- Reduced deformable dense memory usage by avoiding dense posterior storage when only posterior statistics are required.
- Improved atlas performance through direct sparse-stat accumulation, cached deformation buffers, blocked dense expectation, and reuse of precomputed squared norms.
- Improved atlas sparse-query throughput by using multi-worker `cKDTree.query(..., workers=-1)`.
- Reduced repeated allocations in atlas M-step by caching diagonal indices, deformation buffers, and dtype-aligned work arrays.

### Testing

- Added regression tests for blocked dense posterior statistics, atlas float32 internals, deformable float32 defaults, constrained deformable float32 defaults, and float32-vs-float64 accuracy tolerance checks.
- CI now installs `pytest`, runs the regression suite, performs a smoke import test, and builds source and wheel distributions.

### Notes

- The default dtype change is user-visible. If you need the previous default precision, pass `dtype=np.float64` explicitly.