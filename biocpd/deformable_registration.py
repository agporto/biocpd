from builtins import super
import numpy as np
import numbers
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from ._low_rank import (
    gaussian_kernel_eigendecomposition,
    validate_low_rank_options,
)
from .emregistration import EMRegistration
from .utility import gaussian_kernel


class DeformableRegistration(EMRegistration):
    """
    Deformable registration with optional low-rank approximation.

    This class implements a deformable point cloud registration algorithm based
    on the Expectation-Maximization (EM) framework. It uses a Gaussian kernel
    to model the smooth, non-rigid deformation field. For performance on large
    datasets, it supports both a low-rank approximation of the kernel matrix
    and k-d tree based acceleration of the E-step.

    Attributes
    ----------
    alpha: float
        Regularization weight for the coherence of the deformation.
    beta: float
        Width of the Gaussian kernel, controlling the smoothness of the deformation.
    low_rank: bool
        If True, uses a low-rank approximation for the Gaussian kernel to
        improve performance.
    num_eig: int
        Number of eigenvectors to use in the low-rank approximation.
    low_rank_method: str
        Kernel factorization used in low-rank mode. ``"randomized_svd"``
        preserves the historical behavior. ``"pivoted_cholesky"`` avoids
        constructing the full square kernel.
    low_rank_tolerance: float
        Residual-diagonal tolerance for pivoted Cholesky. A value of zero uses
        only a dtype-scaled numerical stopping threshold.
    use_kdtree: bool
        If True, accelerates the E-step by using a k-d tree for nearest neighbor
        searches.
    radius_mode: bool
        If True, in sparse E-step ignore neighbors beyond a numerically safe radius.
    optimize_similarity: bool
        If True, jointly optimize rotation, translation, and optional uniform
        scale with the non-rigid deformation.
    with_scale: bool
        If True, estimate uniform scale when similarity optimization is enabled.
    w: float
        Outlier weight (0 <= w < 1) forwarded to EM base.
    dtype: numpy dtype
        Floating-point dtype used for internal arrays. Defaults to float32.
    """

    def __init__(self,
                 alpha=None,
                 beta=None,
                 low_rank=True,
                 num_eig=300,
                 use_kdtree=True,
                 k=10,
                 radius_mode=False,
                 w=0.0,
                 low_rank_method="randomized_svd",
                 low_rank_tolerance=0.0,
                 optimize_similarity=False,
                 with_scale=True,
                 *args,
                 **kwargs):
        """
        Initializes the DeformableRegistration object.

        Parameters
        ----------
        alpha: float, optional
            Regularization weight. Must be a positive number. Defaults to 2.0.
        beta: float, optional
            Gaussian kernel width. Must be a positive number. Defaults to 2.0.
        low_rank: bool, optional
            Whether to use low-rank approximation. Defaults to True.
        num_eig: int, optional
            Maximum number of eigenmodes in the low-rank approximation.
            Defaults to 300.
        low_rank_method: str, optional
            ``"randomized_svd"`` constructs the complete kernel before
            compressing it and remains the default for backward compatibility.
            ``"pivoted_cholesky"`` builds a deterministic factor from kernel
            columns and uses O(M * num_eig) memory.
        low_rank_tolerance: float, optional
            Early-stopping tolerance for pivoted Cholesky. Defaults to zero.
        use_kdtree: bool, optional
            Whether to use a k-d tree for acceleration. Defaults to True.
        k: int, optional
            Number of nearest neighbors to query in the k-d tree. Defaults to 10.
        radius_mode: bool, optional
            If True, mask neighbors beyond sigma-derived radius in sparse E-step.
        optimize_similarity: bool, optional
            Whether to jointly optimize a global similarity transform. Defaults
            to False, preserving traditional deformable CPD behavior.
        with_scale: bool, optional
            Whether the similarity transform includes uniform scale. Ignored
            when ``optimize_similarity`` is False. Defaults to True.
        w: float, optional
            Outlier weight in [0,1). Forwarded to EM base.
        dtype: numpy dtype, optional
            Floating-point dtype for internal arrays. Defaults to float32.
        """
        kwargs.setdefault("dtype", np.float32)
        super().__init__(w=w, *args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(f"alpha must be a positive number, but got {alpha}")
        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(f"beta must be a positive number, but got {beta}")

        self.alpha = 2.0 if alpha is None else alpha
        self.beta = 2.0 if beta is None else beta
        self.low_rank = low_rank
        self.optimize_similarity = bool(optimize_similarity)
        self.with_scale = bool(with_scale)
        self.R = np.eye(self.D, dtype=self.dtype)
        self.s = self.dtype.type(1.0)
        self.t = np.zeros((1, self.D), dtype=self.dtype)
        self.W = np.zeros((self.M, self.D), dtype=self.dtype)
        self._low_rank_coefficients = None
        self._low_rank_coefficients_current = False
        centered_target = self.X - np.mean(self.X, axis=0)
        target_variance = float(
            np.mean(np.sum(centered_target * centered_target, axis=1))
            / self.D
        )
        self._variance_floor = max(
            np.finfo(self.dtype).eps * target_variance,
            np.finfo(self.dtype).tiny,
        )

        # Pre-compute Gaussian kernel or its low-rank approximation.
        if self.low_rank:
            self.num_eig, self.low_rank_method, self.low_rank_tolerance = (
                validate_low_rank_options(
                    num_eig,
                    low_rank_method,
                    low_rank_tolerance,
                )
            )
            self.Q, self._S_values, self.low_rank_diagnostics = (
                gaussian_kernel_eigendecomposition(
                    self.Y,
                    self.beta,
                    self.num_eig,
                    method=self.low_rank_method,
                    tolerance=self.low_rank_tolerance,
                )
            )
            self._inv_S_values = 1.0 / self._S_values
            self._sqrt_low_rank_weights = np.empty(
                self.M,
                dtype=self.dtype,
            )
            self._weighted_low_rank_basis = np.empty_like(self.Q)
            self._low_rank_projection = np.empty(
                (self.M, self.D),
                dtype=self.dtype,
            )
            # Preserve the historical public factor representation and return
            # contract while using vector forms for internal arithmetic.
            self.S = np.diag(self._S_values)
            self.inv_S = np.diag(self._inv_S_values)
        else:
            self.num_eig = num_eig
            self.low_rank_method = low_rank_method
            self.low_rank_tolerance = low_rank_tolerance
            self.low_rank_diagnostics = None
            self.G = gaussian_kernel(self.Y, self.beta)

        # K-D tree setup for accelerated E-step.
        self.use_kdtree = bool(use_kdtree)
        self.radius_mode = bool(radius_mode)
        if self.use_kdtree:
            self.k = max(1, min(int(k), self.N))
            self.kdtree = cKDTree(self.X)

    def expectation(self):
        """Compute E-step; optionally k-NN sparse with radius gating."""
        if self.use_kdtree:
            distances, indices = self.kdtree.query(self.TY, k=self.k)
            if distances.ndim == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            distances = np.asarray(distances, dtype=self.dtype, order="C")
            indices = np.asarray(indices, dtype=np.int64, order="C")
            distances = np.clip(distances, np.finfo(self.dtype).eps, None)

            # mask invalid neighbors
            mask = np.isfinite(distances)
            # Optional radius gating
            if self.radius_mode:
                rad = float(np.sqrt(-2.0 * self.sigma2 * np.log(np.finfo(self.dtype).eps)))
                mask = mask & (distances <= rad)

            if not mask.any():
                # fall back to dense if no valid sparse candidates
                return super().expectation()

            P_vals = np.exp(
                -(
                    np.minimum(
                        distances,
                        np.sqrt(-2.0 * self.sigma2 * np.log(np.finfo(self.dtype).eps)),
                    ) ** 2
                )
                / (2 * self.sigma2)
            )

            rows = np.arange(self.M).repeat(self.k)
            rows = rows[mask.ravel()]
            cols = indices.ravel()[mask.ravel()]
            vals = P_vals.ravel()[mask.ravel()]

            # Ensure valid column ids
            if cols.size:
                mx = int(cols.max()); mn = int(cols.min())
                if not (0 <= mn and mx < self.N):
                    keep = (cols >= 0) & (cols < self.N)
                    rows, cols, vals = rows[keep], cols[keep], vals[keep]

            P_sparse = csr_matrix((vals, (rows, cols)), shape=(self.M, self.N))

            # Outlier term and column normalization
            c_term = (2 * np.pi * self.sigma2)**(self.D / 2) * self.w / (1 - self.w) * self.M / self.N
            den_col = np.array(P_sparse.sum(axis=0), dtype=self.dtype).ravel() + c_term
            den_col = np.clip(den_col, np.finfo(self.dtype).eps, None)
            inv_den_col = 1.0 / den_col

            self.P = P_sparse.multiply(inv_den_col[np.newaxis, :])
            self.Pt1 = np.array(self.P.sum(axis=0)).ravel()
            self.P1 = np.array(self.P.sum(axis=1)).ravel()
            self.Np = self.P1.sum()
            self.PX = self.P @ self.X
        else:
            self._compute_dense_posterior_stats(store_p=False)

    def update_transform(self):
        """
        Updates the transformation parameters (M-step).

        This method solves for the deformation field weights `W`. If `low_rank`
        is enabled, it uses the Woodbury matrix identity to solve the linear system
        efficiently. Otherwise, it solves the full-rank system directly.
        """
        if self.optimize_similarity:
            tiny = np.finfo(self.dtype).tiny
            self.R, scale, self.t = self._weighted_similarity_update(
                self._current_deformed_source()
            )
            self.s = self.dtype.type(scale if self.with_scale else 1.0)
            xbar = self.PX / np.maximum(self.P1, tiny)[:, None]
            local_targets = self._inverse_similarity(xbar)
            F = self.P1[:, None] * (local_targets - self.Y)
            scale = self.s if self.with_scale else 1.0
            lambda_val = self.dtype.type(
                self.alpha * self.sigma2 / max(scale * scale, tiny)
            )
        else:
            F = self.PX - (self.P1[:, None] * self.Y)
            lambda_val = (
                self.dtype.type(self.alpha * self.sigma2)
                if self.low_rank
                else self.alpha * self.sigma2
            )

        if not self.low_rank:
            A = (self.P1[:, None] * self.G) + lambda_val * np.eye(self.M, dtype=self.dtype)
            B = F
            self.W = np.linalg.solve(A, B)
            self._low_rank_coefficients = None
            self._low_rank_coefficients_current = False
        else:
            # Efficiently solve for W using the Woodbury matrix identity.
            # The system is (diag(P1)QSQ' + lambda*I)W = F, where lambda = alpha*sigma^2.
            # The solution is W = (1/lambda) * (F - DQ(lambda*S^-1 + Q'DQ)^-1 * Q'F).
            # The following code implements this solution.
            self._update_low_rank_transform(self.P1, F, lambda_val)

        if self.optimize_similarity:
            self.R, scale, self.t = self._weighted_similarity_update(
                self._current_deformed_source()
            )
            self.s = self.dtype.type(scale if self.with_scale else 1.0)

    def _apply_similarity(self, points):
        return self.s * (points @ self.R.T) + self.t

    def _inverse_similarity(self, points):
        scale = self.s if self.with_scale else 1.0
        scale = max(float(scale), np.finfo(self.dtype).tiny)
        return (points - self.t) @ self.R / scale

    def _current_deformed_source(self):
        if self.low_rank:
            if self._low_rank_coefficients_current:
                coefficients = self._low_rank_coefficients
            else:
                coefficients = self._S_values[:, None] * (
                    self.Q.T @ self.W
                )
            return self.Y + self.Q @ coefficients
        return self.Y + self.G @ self.W

    def _weighted_similarity_update(self, deformed_source):
        tiny = np.finfo(self.dtype).tiny
        if not np.isfinite(self.Np) or self.Np <= tiny:
            return self.R, float(self.s), self.t
        weight_sum = max(float(self.Np), tiny)
        source_mean = (
            self.P1 @ deformed_source
        ).reshape(1, self.D) / weight_sum
        target_mean = np.sum(
            self.PX, axis=0, keepdims=True
        ) / weight_sum
        covariance = (
            deformed_source.T @ self.PX
            - weight_sum * (source_mean.T @ target_mean)
        )
        U, _, Vt = np.linalg.svd(
            covariance, full_matrices=False
        )
        correction = np.eye(self.D, dtype=self.dtype)
        correction[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        auxiliary_rotation = U @ correction @ Vt
        rotation = auxiliary_rotation.T
        if self.with_scale:
            numerator = np.trace(auxiliary_rotation.T @ covariance)
            denominator = (
                self.P1 @ np.sum(
                    deformed_source * deformed_source, axis=1
                )
                - weight_sum * np.sum(source_mean * source_mean)
            )
            if denominator <= tiny or not np.isfinite(denominator):
                scale = float(self.s)
            else:
                scale = float(numerator / denominator)
                if not np.isfinite(scale) or scale <= tiny:
                    scale = float(self.s)
        else:
            scale = 1.0
        translation = target_mean - scale * (
            source_mean @ rotation.T
        )
        return rotation, scale, translation

    def _update_low_rank_transform(self, weights, F, lambda_val):
        """Solve a low-rank CPD M-step and cache deformation coefficients."""
        # Q' diag(weights) Q is a Gram matrix.  Expressing it through the
        # square-root weighted basis lets optimized BLAS use its A.T @ A path
        # and guarantees a positive-semidefinite system up to roundoff.
        np.sqrt(weights, out=self._sqrt_low_rank_weights)
        np.multiply(
            self.Q,
            self._sqrt_low_rank_weights[:, np.newaxis],
            out=self._weighted_low_rank_basis,
        )
        system = (
            self._weighted_low_rank_basis.T
            @ self._weighted_low_rank_basis
        )
        diagonal = np.diag_indices_from(system)
        system[diagonal] += lambda_val * self._inv_S_values
        rhs = self.Q.T @ F

        try:
            factor = cho_factor(
                system,
                overwrite_a=True,
                check_finite=False,
            )
            coefficients = cho_solve(
                factor,
                rhs,
                overwrite_b=True,
                check_finite=False,
            )
        except np.linalg.LinAlgError:
            # Rebuild after the failed in-place factorization and add a
            # scale-aware diagonal perturbation.  The analytical system is
            # positive definite; this only protects degenerate finite-precision
            # factorizations without changing the ordinary solve.
            system = (
                self._weighted_low_rank_basis.T
                @ self._weighted_low_rank_basis
            )
            system[diagonal] += lambda_val * self._inv_S_values
            diagonal_scale = max(
                float(np.max(np.abs(np.diag(system)))),
                1.0,
            )
            system[diagonal] += max(
                1e-8,
                np.finfo(self.dtype).eps * diagonal_scale,
            )
            factor = cho_factor(
                system,
                overwrite_a=True,
                check_finite=False,
            )
            coefficients = cho_solve(
                factor,
                rhs,
                overwrite_b=True,
                check_finite=False,
            )

        self._low_rank_coefficients = coefficients
        self._low_rank_coefficients_current = True
        np.matmul(
            self.Q,
            coefficients,
            out=self._low_rank_projection,
        )
        np.multiply(
            weights[:, np.newaxis],
            self._low_rank_projection,
            out=self._low_rank_projection,
        )
        np.subtract(F, self._low_rank_projection, out=self.W)
        self.W /= lambda_val

    def transform_point_cloud(self, Y=None):
        """
        Applies the learned deformation to a point cloud.

        Parameters
        ----------
        Y: np.ndarray, optional
            A point cloud to transform. If None, the original source point cloud
            `self.Y` is transformed.

        Returns
        -------
        np.ndarray
            The transformed point cloud.
        """
        if Y is not None:
            # Apply the transformation to a new point cloud.
            G_new = gaussian_kernel(X=Y, Y=self.Y, beta=self.beta)
            deformed = Y + G_new @ self.W
            return (
                self._apply_similarity(deformed)
                if self.optimize_similarity
                else deformed
            )
        else:
            # Transform the original source point cloud.
            if self.low_rank:
                if self._low_rank_coefficients_current:
                    coefficients = self._low_rank_coefficients
                else:
                    # Derive coefficients from W so callers that explicitly
                    # replace the public W array retain historical behavior.
                    coefficients = self._S_values[:, None] * (
                        self.Q.T @ self.W
                    )
                self._low_rank_coefficients = coefficients
                self._low_rank_coefficients_current = False
                deformed = self.Y + self.Q @ coefficients
            else:
                deformed = self.Y + self.G @ self.W
            self.TY = (
                self._apply_similarity(deformed)
                if self.optimize_similarity
                else deformed
            )
            return self.TY

    def update_variance(self):
        """
        Updates the variance of the GMM (M-step).

        Calculates the new `sigma2` based on the current correspondences and
        point positions, and computes the change `diff` from the previous iteration.
        """
        qprev = self.sigma2

        # Corresponding terms from the Q-function.
        xPx = self.Pt1 @ np.sum(self.X**2, axis=1)
        yPy = self.P1 @ np.sum(self.TY**2, axis=1)
        trPXY = np.sum(self.TY * self.PX)

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if not np.isfinite(self.sigma2) or self.sigma2 <= 0:
            self.sigma2 = max(
                self.tolerance / 10.0,
                self._variance_floor,
            )
        self.diff = abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Returns the learned registration parameters.

        Returns
        -------
        tuple
            If not `low_rank`, returns the full Gaussian kernel (`G`) and weights (`W`).
            If `low_rank`, returns the low-rank components (`Q`, `S`) and weights (`W`).
        """
        if self.low_rank:
            return self.Q, self.S, self.W
        else:
            return self.G, self.W

    def get_similarity_parameters(self):
        """
        Returns the learned global similarity transformation.

        Returns
        -------
        tuple
            Rotation matrix, uniform scale, and translation vector
            ``(R, s, t)``. Identity parameters are returned when similarity
            optimization is disabled.
        """
        return self.R, self.s, self.t
