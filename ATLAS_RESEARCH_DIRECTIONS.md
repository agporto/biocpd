# Atlas Registration Research Directions

This note evaluates research and product directions for `AtlasRegistration` after
the current uncertainty-focused work.  The latest implementation adds a local
Laplace-style posterior around atlas coefficients, exposes coefficient
precision/covariance, projects coefficient uncertainty to pointwise variance,
and returns scalar diagnostics.  The remaining ideas below are ranked by
expected value, implementation risk, and fit with the current NumPy/SciPy CPD
architecture.

## Current baseline

`AtlasRegistration` is currently a statistical-shape-model CPD variant.  It
optimizes soft correspondences, atlas coefficients, optional similarity
parameters, and CPD variance.  The model is essentially

\[
T(Y) = s \left((Y + U b) R^T\right) + t,
\]

with an eigenvalue-weighted prior on the coefficient vector `b`.  The current
uncertainty addition approximates the local posterior over `b` by inverting the
same weighted coefficient system used by the atlas M-step.

This means the strongest near-term opportunities are those that reuse the
existing sufficient statistics and coefficient-space solve rather than replacing
the whole registration algorithm.

## Evaluation criteria

Each idea is scored qualitatively on:

- **Impact**: likely scientific or user-facing value.
- **Fit**: how naturally the idea fits the current code structure.
- **Validation burden**: how much numerical evidence is needed before the
  feature can be trusted.
- **Implementation risk**: chance of destabilizing existing registration paths.
- **Recommended timing**: whether it should be done now, later, or deferred.

## Priority summary

| Rank | Idea | Impact | Fit | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | Strong numerical validation for uncertainty | High | Excellent | Low | Do now |
| 2 | Hybrid atlas bases / GP-style residual modes | High | Excellent | Medium | Do next |
| 3 | Correspondence backend abstraction | High | Good | Medium | Do before new matching models |
| 4 | Partial or unbalanced OT correspondence | High | Medium | High | Prototype after backend abstraction |
| 5 | Surface-aware likelihoods | Medium-High | Medium | Medium | Do if surface data are central |
| 6 | Multi-atlas selection or mixtures | Medium | Good | Medium | Later |
| 7 | Diffeomorphic registration | High | Low-Medium | High | Defer |
| 8 | Learned nonlinear priors | High | Low | High | Defer until datasets/training infra exist |
| 9 | Pathology/anomaly decomposition | High | Medium | High | Later, after basis/uncertainty validation |

## 1. Strong numerical validation for uncertainty

**Recommendation: implement immediately.**

The current uncertainty layer is mathematically plausible, but it should be
backed by numerical tests that show it behaves sensibly under known synthetic
conditions.  Shape checks and positive-definiteness checks are useful, but they
are not enough to demonstrate quality.

### Proposed tests

1. **Known coefficient recovery**
   - Generate a synthetic mean shape and basis.
   - Choose `b_true`.
   - Generate `X = Y + U b_true + noise`.
   - Run atlas registration without similarity.
   - Assert recovered coefficients and transformed points are close.

2. **Uncertainty shrinks with more support**
   - Run otherwise identical problems with increasing target support or lower
     observation noise.
   - Assert coefficient posterior trace and mean pointwise variance decrease.

3. **Uncertainty grows with missing observations**
   - Remove a region of target points.
   - Assert pointwise variance is higher near basis-supported missing regions.

4. **Eigenvalue sensitivity**
   - Hold basis fixed and vary eigenvalues.
   - Assert modes with larger prior variance produce larger posterior variance
     when data support is weak.

5. **Calibration smoke test**
   - Repeatedly sample noisy targets from known `b_true`.
   - Compare empirical coefficient error against predicted posterior standard
     deviations.

### Why this is first

It validates the feature that already exists, helps avoid overclaiming, and
creates a benchmark harness for future atlas-prior work.

## 2. Hybrid atlas bases / GP-style residual modes

**Recommendation: implement next.**

The current atlas model already accepts a finite basis `U` and eigenvalues.
That makes hybrid bases a natural extension: concatenate multiple basis families
and use their eigenvalues or penalties as priors.

Examples:

- PCA/SSM population modes.
- Local radial basis modes for residual shape variation.
- Graph Laplacian or spectral modes for smooth geometric residuals.
- Region-specific modes with different regularization strengths.
- Approximate Gaussian-process/Nyström modes.

This direction is inspired by Gaussian Process Morphable Models, where
classical PCA shape models are generalized with continuous Gaussian-process
priors.  In this repository, a practical first step does not need a full GP
implementation; it can expose utilities for constructing and concatenating
finite-rank basis blocks.

### Minimal useful API

```python
U, eigenvalues, groups = compose_atlas_basis(
    pca=(U_pca, L_pca),
    residual=(U_residual, L_residual),
)
```

or a small class:

```python
basis = AtlasBasis()
basis.add_component("pca", U_pca, L_pca)
basis.add_component("local", U_local, L_local)
U, L = basis.as_arrays()
```

### Why this is promising

- Reuses the existing coefficient solve.
- Reuses the new uncertainty approximation.
- Expands modeling power without changing the EM loop.
- Enables local residual modeling while preserving interpretable PCA modes.

### Main validation needs

- Verify concatenated bases recover known mixed coefficients.
- Verify grouped priors affect posterior variance as expected.
- Verify the old single-basis API remains unchanged.

## 3. Correspondence backend abstraction

**Recommendation: implement before adding alternative matching models.**

Right now, dense CPD and sparse k-NN CPD correspondence logic are embedded in
`AtlasRegistration.expectation()`.  Before adding optimal transport, feature
matching, or surface-aware likelihoods, the correspondence computation should be
made pluggable.

### Desired interface

Each backend should return the same sufficient statistics consumed by the atlas
M-step:

```python
P1, Pt1, PX, Np, optional_P = backend.compute(TY, X, sigma2)
```

The existing dense and sparse paths can become the first two backends.

### Why this is valuable

- Makes future experiments safer.
- Separates matching assumptions from atlas coefficient optimization.
- Allows exact dense, sparse approximate, OT, feature-aware, and surface-aware
  correspondence models to share the same M-step.

### Main risk

Refactoring correspondence code can break existing dense/sparse behavior, so it
needs equivalence tests against the current implementation.

## 4. Partial or unbalanced optimal transport correspondence

**Recommendation: prototype after the backend abstraction.**

Partial and unbalanced optimal transport are attractive because biological and
medical point clouds often have missing structures, outliers, or density
mismatch.  CPD's outlier term helps, but it does not fully model unmatched mass
on both sides.

An OT backend could replace the CPD posterior matrix with a transport plan and
then compute equivalent atlas sufficient statistics:

\[
P1_i = \sum_j \pi_{ij}, \qquad PX_i = \sum_j \pi_{ij} X_j.
\]

### Why not implement first

A robust OT implementation introduces significant design choices:

- entropic regularization strength,
- mass relaxation penalties,
- stopping criteria,
- numerical stabilization,
- dense versus sparse costs,
- behavior under partial overlap.

It is better to isolate this behind a backend interface first.

### First prototype scope

- Dense small-problem Sinkhorn backend.
- No sparse acceleration initially.
- Synthetic tests with known missing target regions.
- Compare against dense CPD on complete data.

## 5. Surface-aware likelihoods

**Recommendation: implement only if mesh/surface data are central.**

The current likelihood is isotropic point-to-point Gaussian.  For surfaces,
point-to-plane or anisotropic covariance models can be more appropriate:

\[
d_{ij}^2 = (x_j - T(y_i))^T \Sigma_i^{-1} (x_j - T(y_i)).
\]

Here `Sigma_i` could encode surface-normal and tangent uncertainty.

### Benefits

- Better behavior on smooth surfaces.
- Less tangential over-penalization.
- More ICP-like local accuracy while retaining soft correspondences.

### Costs

- Requires normals or mesh connectivity.
- Adds more geometry assumptions to a point-cloud package.
- Needs careful tests for orientation, normal flips, and degenerate normals.

## 6. Multi-atlas selection or mixtures

**Recommendation: later.**

A multi-atlas wrapper is useful when data contain subpopulations.  A simple first
version could run several atlases and choose the best by residual, coefficient
Mahalanobis norm, or approximate objective.

A full mixture model would infer both atlas class and registration parameters:

\[
p(Y) = \sum_c \pi_c \mathcal{N}(\mu_c, U_c \Lambda_c U_c^T).
\]

### Why later

This is valuable, but less foundational than validating uncertainty and
expanding the basis model.  It also needs a reliable scoring/evidence metric,
which the current atlas implementation does not yet expose as a calibrated
objective.

## 7. Diffeomorphic registration

**Recommendation: defer.**

Diffeomorphic registration is mathematically attractive for topology-preserving
anatomical deformation, but it is a substantial change in model class.  The
atlas would need to parameterize a velocity field or flow rather than direct
point displacement.

### Why defer

- Requires flow integration or LDDMM-style machinery.
- Requires Jacobian/topology diagnostics.
- Changes transformation semantics.
- Adds a much larger validation burden.

This should be treated as a separate research branch, not a quick next feature.

## 8. Learned nonlinear priors

**Recommendation: defer until there is training infrastructure.**

VAE, normalizing-flow, or diffusion shape priors could model nonlinear
population variation, but they would fundamentally change the package's
requirements and workflow.

### Costs

- Requires datasets and training loops.
- Adds major dependencies such as PyTorch or JAX.
- Requires serialization, reproducibility, and model-versioning decisions.
- Makes tests and examples heavier.

A more practical near-term route is hybrid finite bases first, then learned
priors later if there is clear demand.

## 9. Pathology or anomaly decomposition

**Recommendation: later, after uncertainty and hybrid bases.**

A useful biological model would decompose observations into normal atlas
variation plus local abnormality:

\[
X \approx T(\mu + U b + a),
\]

where `U b` is normal population variation and `a` is sparse or localized
abnormal deformation.

### Why promising

- Could identify lesions, defects, missing regions, or local shape changes.
- Makes atlas registration useful for downstream biological interpretation.

### Why later

It needs reliable uncertainty and a flexible residual basis first.  Otherwise,
normal variation, registration error, and anomaly signal will be hard to
separate.

## Recommended roadmap

### Phase 1: Make the uncertainty feature trustworthy

- Add synthetic coefficient-recovery tests.
- Add uncertainty shrinkage/growth tests.
- Add calibration smoke tests.
- Add a small benchmark/demo script that prints coefficient error, transformed
  RMSE, posterior trace, and mean/max pointwise variance.

### Phase 2: Add hybrid finite basis support

- Implement basis composition utilities.
- Support grouped basis metadata.
- Test mixed PCA + residual recovery.
- Ensure old `U`/`eigenvalues` usage remains unchanged.

### Phase 3: Refactor correspondences into backends

- Extract dense CPD backend.
- Extract sparse k-NN CPD backend.
- Prove equivalence to current outputs on fixed synthetic inputs.

### Phase 4: Prototype partial/unbalanced OT

- Add dense small-problem OT backend.
- Validate on partial-overlap synthetic cases.
- Compare to dense CPD and sparse CPD.

### Phase 5: Consider larger research features

- Surface-aware likelihoods if data are meshes/surfaces.
- Multi-atlas wrappers if there are clear subpopulations.
- Diffeomorphic registration only as a dedicated research branch.
- Learned priors only after training infrastructure exists.

## Bottom line

The highest-value next step is not to add a new ambitious model immediately. It
is to **numerically prove the uncertainty feature**, then extend the atlas prior
through **hybrid finite bases**.  Those two steps build directly on the current
architecture and create a stronger foundation for later OT, surface-aware,
multi-atlas, or diffeomorphic work.
