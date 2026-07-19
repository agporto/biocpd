# biocpd

Coherent Point Drift (CPD) registration in pure NumPy/SciPy with fast variants:
- Rigid and Affine CPD
- Deformable CPD with low-rank (randomized SVD) and k-d tree accelerated E-step
- Constrained Deformable CPD with correspondence priors
- Atlas/SSM-based CPD (`AtlasRegistration`) optimized in coefficient space
- Opt-in pose-marginalized initialization for severely misaligned atlas inputs

## Why biocpd?
- Fast: sparse k-NN E-step, low-rank kernels, and efficient linear solvers
- Flexible: rigid, affine, unconstrained and constrained deformable, and SSM/atlas-based
- Simple: pure NumPy/SciPy implementation; easy to read and extend

## Install

```bash
pip install -r requirements.txt
# optional (recommended for building)
pip install build wheel
```

## Build wheel

```bash
# From repository root
python -m build
# or legacy
python setup.py sdist bdist_wheel
```

## Quickstart

```python
import numpy as np
from biocpd import RigidRegistration, AffineRegistration, DeformableRegistration, ConstrainedDeformableRegistration, AtlasRegistration

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))           # target
Y = X + 0.05 * rng.normal(size=(200,3)) # source (noisy)

# Rigid CPD
rig = RigidRegistration(X=X, Y=Y, max_iterations=50, use_kdtree=True, k=10)
TY_rigid, (s, R, t) = rig.register()

# Affine CPD
aff = AffineRegistration(X=X, Y=Y, max_iterations=50, use_kdtree=True, k=10)
TY_affine, (B, t) = aff.register()

# Deformable CPD (scalable low-rank kernel + k-d tree)
defm = DeformableRegistration(X=X, Y=Y, alpha=2.0, beta=2.0, low_rank=True, num_eig=80,
                              low_rank_method="pivoted_cholesky",
                              use_kdtree=True, k=10, radius_mode=False, w=0.05,
                              max_iterations=50)
TY_def, params = defm.register()

# Constrained Deformable CPD
ids = np.arange(10)
con = ConstrainedDeformableRegistration(X=X, Y=Y, alpha=2.0, beta=2.0, low_rank=True, num_eig=80,
                                        low_rank_method="pivoted_cholesky",
                                        use_kdtree=True, k=10, e_alpha=1e-4,
                                        source_id=ids, target_id=ids,
                                        max_iterations=50)
TY_con, params_con = con.register()

# Atlas / Statistical Shape Model CPD
M, D, K = 200, 3, 12
mean_shape = rng.normal(size=(M, D))
U = rng.normal(size=(M*D, K))
L = np.abs(rng.normal(size=(K,))) + 1e-1
atl = AtlasRegistration(X=X, Y=mean_shape, mean_shape=None,
                        U=U, eigenvalues=L, lambda_reg=0.1,
                        normalize=True, use_kdtree=True, k=10, radius_mode=False,
                        optimize_similarity=True, with_scale=True, w=0.02,
                        max_iterations=50)
TY_atl, params_atl = atl.register()
```

## Pose-marginalized atlas initialization

`pose_marginalized_initialization` evaluates a deterministic lattice of global
rotation hypotheses, jointly refines pose and shape for the strongest
hypotheses, and returns a warm-start state for `AtlasRegistration`. It is
strictly opt-in: existing `AtlasRegistration` construction and registration
behavior are unchanged.

```python
from biocpd import (
    AtlasRegistration,
    PoseMarginalizedConfig,
)

pose_config = PoseMarginalizedConfig(seed=0)
initial = pose_config.initialize(mean_shape, X, U, L)

atl = AtlasRegistration(
    X=X,
    Y=mean_shape,
    mean_shape=None,
    U=U,
    eigenvalues=L,
    lambda_reg=0.1,
    normalize=True,
    optimize_similarity=True,
    with_scale=True,
)
atl.set_initial_state(
    initial.coefficients,
    initial.rotation,
    initial.scale,
    initial.translation,
    world_units=True,
)
TY_atl, params_atl = atl.register()
```

The function API remains available for one-off configuration:

```python
from biocpd import pose_marginalized_initialization

initial = pose_marginalized_initialization(
    mean_shape,
    X,
    U,
    L,
    rotation_count=193,
    seed=0,
)
```

`rotation_count` is the exact total hypothesis budget, including identity.
The pose coefficient regularization and outlier defaults are
`lambda_reg=0.1` and `outlier_weight=0.05`, matching the validated real-data
configuration.
By default, every coarse hypothesis receives all eight coarse EM iterations
(`coarse_screen_iterations=coarse_iterations=8` and
`coarse_survivor_count=rotation_count=193`). Set a smaller screen iteration
count and survivor count to opt into staged pruning. Refinement uses the full
source model by default (`refine_source_count=None`) and at most
`refine_target_count` target points; every finalist is scored against the
complete source model. Coarse hypotheses use the E-step trajectory objective
by default (`coarse_score_mode="trajectory"`), which is less sensitive to a
single final coordinate update on symmetric shapes; refined finalists are
always ranked with the exact full-source likelihood. Set
`coarse_score_mode="final"` to recover final-state coarse scoring. Set
`n_jobs` above 1 (or to -1 for all detected CPUs) to evaluate independent
hypotheses concurrently.

Pose initialization adds computation before the final atlas registration. Use
it when global orientation is uncertain or severe misalignment is expected;
skip it for inputs already known to be aligned.

## Key options
- `use_kdtree`, `k`: enable sparse E-step for speed on large data
- `low_rank`, `num_eig` (deformable): low-rank kernel for fast M-step
- `low_rank_method` (deformable): `"randomized_svd"` preserves the historical
  default; `"pivoted_cholesky"` avoids constructing the full square kernel and
  is intended for large point sets
- `low_rank_tolerance` (deformable): optional residual-diagonal stopping
  tolerance for pivoted Cholesky; the default `0.0` uses the requested rank
  unless the kernel becomes numerically rank deficient
- Low-rank deformable M-steps form their weighted coefficient system as a
  square-root-weighted Gram matrix. This is exact, applies to constrained and
  unconstrained registration, and requires no additional option.
- `radius_mode`: optional radius gating in sparse E-step (off by default)
- `w`: outlier weight (0 ≤ w < 1) in GMM
- `dtype` (deformable, constrained deformable, atlas): defaults to `np.float32`; set `dtype=np.float64` when you need the extra precision
- `dense_block_size` (atlas): defaults to a cache-aware block selected from the
  source and target sizes; pass a positive integer to force a specific block
- `coefficient_solver` (atlas): `"cholesky"` preserves the historical direct
  solve, `"cg"` uses a warm-started matrix-free solve for atlases with many
  modes, and `"auto"` selects CG at `coefficient_auto_threshold` modes
- `coefficient_tolerance`, `coefficient_max_iterations` (atlas): control the CG
  residual target and iteration budget; failures fall back to Cholesky and are
  reported through `coefficient_solver_diagnostics`
- `normalize` (atlas): improves stability across scales
- `mean_shape` (atlas): with `normalize=True`, pass `mean_shape` as `(M, D)`

## Acknowledgements
- This work builds on the excellent original CPD implementation by Siavash Khallaghi and Anthony Gatti (`pycpd`, MIT-licensed) and the CPD method by Myronenko and Song.
- Repository for `pycpd`: https://github.com/siavashk/pycpd

## Citation
If you use this package in academic work, please cite CPD:

- Myronenko, A. and Song, X., "Point Set Registration: Coherent Point Drift," in IEEE Transactions on Pattern Analysis and Machine Intelligence, 2010.

## License
MIT 
