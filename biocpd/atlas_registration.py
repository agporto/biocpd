import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, Callable, Dict, Any
from .emregistration import EMRegistration


class AtlasRegistration(EMRegistration):
    """
    Statistical-shape-model (SSM) based CPD registration with optional similarity.

    Parameters
    ----------
    X : np.ndarray
        Target point cloud of shape (N, D).
    Y : np.ndarray
        Source point cloud of shape (M, D). If `mean_shape` is provided, the
        registration is performed from the mean shape instead of the raw `Y`.
    mean_shape : Optional[np.ndarray]
        Mean shape as (M, D). If provided, replaces `Y` as the initial shape in
        the model's space. With `normalize=True`, only (M, D) is accepted.
    U : np.ndarray
        Shape basis. Either (M, D, K) or (M*D, K).
    eigenvalues : Optional[np.ndarray]
        Eigenvalues for each basis mode, shape (K,).
    lambda_reg : float
        Regularization weight for the coefficient prior (default 0.1).
    normalize : bool
        If True, normalize inputs to zero-mean/unit-scale for robustness.
    use_kdtree : bool
        If True, enable k-NN accelerated E-step.
    k : int
        Number of neighbors for the k-NN E-step.
    kdtree_radius_scale : float
        Threshold scale to switch to sparse mode based on sigma2.
    dense_block_size : Optional[int]
        Dense E-step column block size. By default, Atlas selects a cache-aware
        block from the source and target sizes.
    store_posterior : bool
        If True, retain the posterior matrix `P`. Keeping this disabled reduces
        memory pressure for large atlas problems.
    optimize_similarity : bool
        If True, optimize similarity (R, s, t) each iteration via weighted Procrustes.
    with_scale : bool
        If True, include uniform scale in similarity.
    radius_mode : bool
        If True, in sparse E-step ignore neighbors with distance greater than a
        numerically safe radius tied to current sigma2.
    coefficient_solver : {"cholesky", "cg", "auto"}
        Solver for the weighted shape-coefficient system. Cholesky preserves
        the historical direct solve. CG is matrix-free and warm-started;
        ``auto`` selects it when the number of modes reaches
        ``coefficient_auto_threshold``.
    coefficient_tolerance : Optional[float]
        Relative residual tolerance for CG. Defaults to a dtype-aware value.
    coefficient_max_iterations : Optional[int]
        Maximum CG iterations. Defaults to the number of shape modes.
    coefficient_auto_threshold : int
        Mode-count threshold used by ``coefficient_solver="auto"``.
    w : float
        Outlier weight (0 <= w < 1). Forwarded to EM base.
    Notes
    -----
    This registrar is SSM-based, so both `U` and `eigenvalues` are required.
    **kwargs : Any
        Forwarded to EM base (e.g., max_iterations, tolerance).
    """
    @staticmethod
    def _normalize_point_cloud(pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        pc = np.asarray(pc)
        eps = np.finfo(pc.dtype).eps if np.issubdtype(pc.dtype, np.floating) else np.finfo(np.float64).eps
        c = np.mean(pc, axis=0, dtype=pc.dtype)
        centered = pc - c
        s = np.sqrt(np.mean(np.sum(centered**2, axis=1), dtype=pc.dtype), dtype=pc.dtype)
        scale = s if s >= eps else 1.0
        return centered / scale, c, scale

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 *,
                 mean_shape: Optional[np.ndarray] = None,
                 U: np.ndarray = None,
                 eigenvalues: Optional[np.ndarray] = None,
                 lambda_reg: float = 0.1,
                 normalize: bool = False,
                 use_kdtree: bool = True,
                 k: int = 10,
                 kdtree_radius_scale: float = 10.0,
                 dense_block_size: Optional[int] = None,
                 store_posterior: bool = False,
                 optimize_similarity: bool = True,
                 with_scale: bool = True,
                 radius_mode: bool = False,
                 coefficient_solver: str = "cholesky",
                 coefficient_tolerance: Optional[float] = None,
                 coefficient_max_iterations: Optional[int] = None,
                 coefficient_auto_threshold: int = 128,
                 w: float = 0.0,
                 dtype: np.dtype = np.float32,
                 **kwargs: Any) -> None:
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(f"dtype must be a floating-point type, but got {dtype}")
        if U is None:
            raise ValueError("U must be provided with shape (M, D, K) or (M*D, K).")
        if lambda_reg is None or eigenvalues is None:
            raise ValueError("lambda_reg and eigenvalues must be provided.")
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be non-negative, but got {lambda_reg}")
        if not isinstance(kdtree_radius_scale, (int, float)) or kdtree_radius_scale <= 0:
            raise ValueError(
                f"kdtree_radius_scale must be a positive number, but got {kdtree_radius_scale}"
            )
        if (
            not isinstance(coefficient_solver, str)
            or coefficient_solver not in {"cholesky", "cg", "auto"}
        ):
            raise ValueError(
                "coefficient_solver must be 'cholesky', 'cg', or 'auto'"
            )
        if coefficient_tolerance is not None and (
            not isinstance(
                coefficient_tolerance,
                (int, float, np.integer, np.floating),
            )
            or isinstance(coefficient_tolerance, (bool, np.bool_))
            or not np.isfinite(coefficient_tolerance)
            or coefficient_tolerance <= 0
        ):
            raise ValueError("coefficient_tolerance must be finite and positive")
        if coefficient_max_iterations is not None and (
            not isinstance(coefficient_max_iterations, (int, np.integer))
            or isinstance(coefficient_max_iterations, (bool, np.bool_))
            or coefficient_max_iterations < 1
        ):
            raise ValueError(
                "coefficient_max_iterations must be a positive integer"
            )
        if (
            not isinstance(coefficient_auto_threshold, (int, np.integer))
            or isinstance(coefficient_auto_threshold, (bool, np.bool_))
            or coefficient_auto_threshold < 1
        ):
            raise ValueError(
                "coefficient_auto_threshold must be a positive integer"
            )
        self.normalize = normalize
        X_raw = np.asarray(X, dtype); Y_raw = np.asarray(Y, dtype)
        if normalize:
            Xn, c, s = self._normalize_point_cloud(X_raw); Yn = (Y_raw - c) / s
            self.target_centroid, self.target_scale = c, s; X, Y = Xn, Yn
            if mean_shape is not None:
                ms = np.asarray(mean_shape, dtype)
                if ms.ndim != 2 or ms.shape != Y_raw.shape:
                    raise ValueError(
                        "With normalize=True, mean_shape must be provided as shape (M, D)."
                    )
                mean_shape = (ms - c) / s
            if U is not None: U = np.asarray(U, dtype) / s
        else:
            X, Y = X_raw, Y_raw; self.target_centroid = np.zeros(Y_raw.shape[1], dtype=dtype); self.target_scale = 1.0
        super().__init__(X=X, Y=Y, w=w, dense_block_size=dense_block_size, dtype=dtype, **kwargs)

        self.use_kdtree = use_kdtree; self._use_sparse = False
        self.N, self.D = self.X.shape; self.M = self.Y.shape[0]
        self.lambda_reg = float(lambda_reg); self.optimize_similarity = bool(optimize_similarity); self.with_scale = bool(with_scale)
        self.radius_mode = bool(radius_mode)
        self.store_posterior = bool(store_posterior)
        self.coefficient_solver = coefficient_solver
        self.coefficient_tolerance = (
            float(coefficient_tolerance)
            if coefficient_tolerance is not None
            else max(10.0 * np.finfo(dtype).eps, 1e-10)
        )
        self.coefficient_max_iterations = coefficient_max_iterations
        self.coefficient_auto_threshold = int(coefficient_auto_threshold)

        ev = np.asarray(eigenvalues, dtype)
        if ev.ndim != 1: raise ValueError("eigenvalues has an invalid shape.")
        U_arr = np.asarray(U, dtype); self.K = ev.shape[0]; self.MD = self.M * self.D
        if U_arr.ndim == 3 and U_arr.shape == (self.M, self.D, self.K): self.U_flat = U_arr.reshape(self.MD, self.K)
        elif U_arr.ndim == 2 and U_arr.shape == (self.MD, self.K): self.U_flat = U_arr.copy()
        else: raise ValueError("U has an invalid shape.")

        if mean_shape is not None:
            ms = np.asarray(mean_shape, dtype)
            if ms.ndim == 1 and ms.size == self.MD: self.mean_shape = ms.reshape(self.M, self.D)
            elif ms.ndim == 2 and ms.shape == (self.M, self.D): self.mean_shape = ms.copy()
            else: raise ValueError("mean_shape has an invalid shape.")
            self.Y = self.mean_shape.copy()
        else: self.mean_shape = None

        self.L = ev.copy()
        if self.L.shape != (self.K,): raise ValueError("eigenvalues has an invalid shape.")
        self.L = np.asarray(self.L, dtype=self.dtype)
        floor = max(1e-12, np.finfo(self.dtype).eps)
        self.L = np.clip(self.L, floor, None)
        self.invL = 1.0 / self.L
        self._diag_idx = np.diag_indices(self.K)
        self._log_eps = np.log(np.finfo(self.dtype).eps)

        dx = self.X.max(0) - self.X.min(0)
        self.kdtree_radius_threshold = float(np.linalg.norm(dx)) / float(kdtree_radius_scale)
        self._variance_floor = 1e-6 * float(np.dot(dx, dx)) / self.D
        self.k = max(1, min(int(k), self.N))
        if use_kdtree: self.kdtree = cKDTree(self.X)

        self.r = np.sqrt(self.sigma2)
        self.R = np.eye(self.D, dtype=self.dtype)
        self.s = self.dtype.type(1.0)
        self.t = np.zeros((1, self.D), dtype=self.dtype)
        self.TY = self.Y.copy(); self.TY_world = self._denormalize(self.TY)

        self.P1_ext = np.empty(self.MD, dtype=self.dtype); self.WU = np.empty((self.MD, self.K), dtype=self.dtype)
        self.b = np.zeros((self.K, 1), dtype=self.dtype); self.prev_b = self.b.copy()
        self._deformation = np.zeros((self.M, self.D), dtype=self.dtype)
        self._X_augmented = np.empty(
            (self.N, self.D + 1),
            dtype=self.dtype,
        )
        self._X_augmented[:, 0] = 1
        self._X_augmented[:, 1:] = self.X
        self._basis_norm_squared = None
        self.coefficient_solver_diagnostics = {
            "requested_method": self.coefficient_solver,
            "last_method": None,
            "solve_count": 0,
            "cg_solve_count": 0,
            "fallback_count": 0,
            "last_iterations": 0,
            "last_relative_residual": None,
            "last_fallback_reason": None,
        }

    def _get_dense_block_size(self) -> int:
        """Choose a cache-aware dense E-step block unless explicitly set."""
        if self.dense_block_size is not None:
            return super()._get_dense_block_size()

        itemsize = np.dtype(self.dtype).itemsize
        matrix_bytes = self.M * self.N * itemsize
        if self.N <= 2048 and matrix_bytes <= 32 * 1024**2:
            return self.N

        target_bytes = 3 * 1024**2
        columns = max(target_bytes // max(self.M * itemsize, 1), 1)
        power_of_two = 1 << (int(columns).bit_length() - 1)
        return min(self.N, max(32, min(power_of_two, 1024)))

    def _compute_dense_posterior_stats(
        self,
        store_p=True,
        TY=None,
        block_size=None,
    ) -> None:
        """Accumulate Atlas posterior moments without redundant reductions."""
        transformed = (
            self.TY
            if TY is None
            else np.asarray(TY, dtype=self.dtype)
        )
        transformed_squared = np.sum(
            transformed * transformed,
            axis=1,
        )[:, None]
        block_size = (
            self._get_dense_block_size()
            if block_size is None
            else max(1, min(int(block_size), self.N))
        )
        outlier = (
            (2 * np.pi * self.sigma2) ** (self.D / 2)
            * self.w
            / (1.0 - self.w)
            * self.M
            / self.N
        )

        posterior = (
            np.empty((self.M, self.N), dtype=self.dtype)
            if store_p
            else None
        )
        target_mass = np.zeros(self.N, dtype=self.dtype)
        source_mass = np.zeros(self.M, dtype=self.dtype)
        source_target_sum = np.zeros((self.M, self.D), dtype=self.dtype)

        for start in range(0, self.N, block_size):
            stop = min(start + block_size, self.N)
            target = self.X[start:stop]
            weights = transformed @ target.T
            weights *= -2.0
            weights += transformed_squared
            weights += self.X_sq[start:stop][None, :]
            np.maximum(weights, 0.0, out=weights)
            weights *= -1.0 / (2.0 * self.sigma2)
            np.exp(weights, out=weights)

            kernel_sum = np.sum(weights, axis=0)
            denominator = kernel_sum + outlier
            np.maximum(denominator, self._tiny, out=denominator)
            target_mass[start:stop] = kernel_sum / denominator
            weights /= denominator[None, :]

            if store_p:
                posterior[:, start:stop] = weights
            moments = weights @ self._X_augmented[start:stop]
            source_mass += moments[:, 0]
            source_target_sum += moments[:, 1:]

        self.P = posterior
        self.Pt1 = target_mass
        self.P1 = source_mass
        self.Np = float(source_mass.sum())
        self.PX = source_target_sum

    def _denormalize(self, pts: np.ndarray) -> np.ndarray:
        return pts*self.target_scale + self.target_centroid if self.normalize else pts
    def _apply_similarity(self, Z: np.ndarray) -> np.ndarray: return (self.s * (Z @ self.R.T)) + self.t
    def _inverse_similarity(self, Z: np.ndarray) -> np.ndarray:
        s = self.s if self.with_scale else 1.0
        s = s if s>np.finfo(self.dtype).tiny else 1.0
        return (Z - self.t) @ self.R / s

    def _require_unstarted(self) -> None:
        if self.iteration != 0:
            raise RuntimeError("initial state must be set before registration starts")

    def set_initial_coefficients(self, coefficients: np.ndarray) -> None:
        """Set shape coefficients before registration starts."""
        self._require_unstarted()
        coefficients = np.asarray(coefficients, dtype=self.dtype)
        if coefficients.size != self.K:
            raise ValueError(
                f"coefficients must contain {self.K} values, got {coefficients.size}"
            )
        coefficients = coefficients.reshape(self.K, 1)
        if not np.isfinite(coefficients).all():
            raise ValueError("initial coefficients must be finite")
        self.b = coefficients.copy()
        self.prev_b = coefficients.copy()
        self._deformation[:] = self.U_flat.dot(self.b).reshape(self.M, self.D)
        self.TY = self._apply_similarity(self.Y + self._deformation)
        self.TY_world = self._denormalize(self.TY)

    def set_initial_similarity(
        self,
        rotation: np.ndarray,
        scale: float,
        translation: np.ndarray,
        *,
        world_units: bool = True,
    ) -> None:
        """Set a similarity transform before registration starts."""
        self._require_unstarted()
        rotation = np.asarray(rotation, dtype=self.dtype)
        translation = np.asarray(translation, dtype=self.dtype).reshape(1, self.D)
        if rotation.shape != (self.D, self.D):
            raise ValueError("rotation has an invalid shape")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("initial similarity must be finite")
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("scale must be finite and positive")
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(self.D),
            atol=1e-5,
        ) or np.linalg.det(rotation) <= 0:
            raise ValueError("rotation must be a proper orthogonal matrix")
        self.R = rotation
        self.s = float(scale if self.with_scale else 1.0)
        if self.normalize and world_units:
            centroid = self.target_centroid.reshape(1, self.D)
            self.t = (
                self.s * (centroid @ self.R.T)
                + translation
                - centroid
            ) / self.target_scale
        else:
            self.t = translation
        self.TY = self._apply_similarity(self.Y + self._deformation)
        self.TY_world = self._denormalize(self.TY)

    def set_initial_state(
        self,
        coefficients: np.ndarray,
        rotation: np.ndarray,
        scale: float,
        translation: np.ndarray,
        *,
        world_units: bool = True,
    ) -> None:
        """Warm-start coefficients and similarity in a consistent order."""
        self.set_initial_coefficients(coefficients)
        self.set_initial_similarity(
            rotation,
            scale,
            translation,
            world_units=world_units,
        )

    def register(self, callback: Callable[..., None] = lambda **kwargs: None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.normalize and callable(callback):
            def _cb(**kw):
                kw = dict(kw)
                if 'X' in kw: kw['X'] = self._denormalize(kw['X'])
                if 'Y' in kw: kw['Y'] = self._denormalize(kw['Y'])
                callback(**kw)
            TY_norm, params = super().register(callback=_cb)
        else:
            TY_norm, params = super().register(callback=callback)
        return self._denormalize(TY_norm), params

    def maximization(self) -> None:
        # Atlas update_transform already computes the new transformed points.
        self.update_transform()
        self.update_variance()

    def _accumulate_sparse_stats(self, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> None:
        c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1 - self.w) * self.M / self.N
        den = np.bincount(cols, weights=weights, minlength=self.N).astype(self.dtype, copy=False)
        den += c
        np.maximum(den, self._tiny, out=den)

        norm_weights = (weights / den[cols]).astype(self.dtype, copy=False)
        self.P = csr_matrix((norm_weights, (rows, cols)), shape=(self.M, self.N)) if self.store_posterior else None
        self.Pt1 = np.bincount(cols, weights=norm_weights, minlength=self.N).astype(self.dtype, copy=False)
        self.P1 = np.bincount(rows, weights=norm_weights, minlength=self.M).astype(self.dtype, copy=False)
        self.Np = float(self.P1.sum())

        PX = np.empty((self.M, self.D), dtype=self.dtype)
        for dim in range(self.D):
            PX[:, dim] = np.bincount(
                rows, weights=norm_weights * self.X[cols, dim], minlength=self.M
            )
        self.PX = PX

    def expectation(self) -> None:
        if self.use_kdtree and not self._use_sparse and self.sigma2 < self.kdtree_radius_threshold ** 2:
            self._use_sparse = True

        if self._use_sparse:
            valid_rows = np.all(np.isfinite(self.TY), axis=1)
            TYq = self.TY[valid_rows] if not valid_rows.all() else self.TY

            if TYq.size == 0:
                self._use_sparse = False
                return self.expectation()

            d, idx = self.kdtree.query(TYq, k=self.k, workers=-1)
            if d.ndim == 1:
                d = d[:, None]
                idx = idx[:, None]
            d = np.asarray(d, dtype=self.dtype, order="C")
            idx = np.asarray(idx, dtype=np.int64, order="C")

            mask = np.isfinite(d) & (idx >= 0) & (idx < self.N)
            # Optional radius gating
            if self.radius_mode:
                rad = float(np.sqrt(-2.0 * self.sigma2 * self._log_eps))
                mask &= (d <= rad)
            if not mask.any():
                self._use_sparse = False
                return self.expectation()

            nz_rows, nz_cols = np.nonzero(mask)
            base_rows = np.where(valid_rows)[0] if not valid_rows.all() else np.arange(self.M)
            rows = base_rows[nz_rows]
            cols = idx[nz_rows, nz_cols]
            vals = np.exp(-(d[nz_rows, nz_cols] ** 2) / (2.0 * self.sigma2))

            self._accumulate_sparse_stats(rows, cols, vals)
            return

        self._compute_dense_posterior_stats(store_p=self.store_posterior)

    def _weighted_similarity_update(self, Yb: np.ndarray, xbar: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray]:
        tiny = np.finfo(self.dtype).tiny
        w = self.P1; wsum = max(self.Np, tiny)
        if xbar is None:
            xbar = self.PX / np.maximum(w[:, None], tiny)
        mu_x = (w[:,None]*xbar).sum(axis=0, keepdims=True)/wsum
        mu_y = (w[:,None]*Yb ).sum(axis=0, keepdims=True)/wsum
        Xc = xbar - mu_x; Yc = Yb - mu_y
        C = (Yc.T * w).dot(Xc)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        M = np.eye(self.D, dtype=self.dtype); M[-1,-1] = np.sign(np.linalg.det(U@Vt))
        A = U @ M @ Vt
        R = A.T
        if self.with_scale:
            num = np.trace(A.T @ C)
            den = float((w * (Yc*Yc).sum(axis=1)).sum()); s = float(num / max(den, tiny))
        else:
            s = self.dtype.type(1.0)
        t = mu_x - s*(mu_y @ R.T)
        return R, s, t

    def _selected_coefficient_solver(self) -> str:
        if self.coefficient_solver == "auto":
            return (
                "cg"
                if self.K >= self.coefficient_auto_threshold
                else "cholesky"
            )
        return self.coefficient_solver

    def _coefficient_system_product(
        self,
        vector: np.ndarray,
        weights: np.ndarray,
        prior_diagonal: np.ndarray,
    ) -> np.ndarray:
        projected = self.U_flat @ vector
        projected *= weights
        return self.U_flat.T @ projected + prior_diagonal * vector

    def _solve_coefficients_cholesky(
        self,
        residual: np.ndarray,
        prior_diagonal: np.ndarray,
    ) -> np.ndarray:
        np.sqrt(self.P1_ext, out=self.P1_ext)
        self.WU[:] = self.U_flat * self.P1_ext[:, None]
        system = self.WU.T.dot(self.WU)
        system[self._diag_idx] += prior_diagonal
        weighted_residual = residual * self.P1_ext
        rhs = self.WU.T.dot(weighted_residual).reshape(self.K, 1)
        try:
            factor, lower = cho_factor(
                system,
                overwrite_a=True,
                check_finite=False,
            )
            return cho_solve(
                (factor, lower),
                rhs,
                overwrite_b=True,
                check_finite=False,
            )
        except np.linalg.LinAlgError:
            system = self.WU.T.dot(self.WU)
            diagonal_scale = max(
                float(np.max(np.abs(np.diag(system)))),
                1.0,
            )
            jitter = max(
                1e-8,
                np.finfo(self.dtype).eps * diagonal_scale,
            )
            system[self._diag_idx] += prior_diagonal + jitter
            factor, lower = cho_factor(
                system,
                overwrite_a=True,
                check_finite=False,
            )
            return cho_solve(
                (factor, lower),
                rhs,
                overwrite_b=True,
                check_finite=False,
            )

    def _solve_coefficients_cg(
        self,
        residual: np.ndarray,
        prior_diagonal: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], int, float, Optional[str]]:
        weights = self.P1_ext
        weighted_residual = residual * weights
        rhs = self.U_flat.T @ weighted_residual
        solution = self.b.reshape(-1).copy()
        system_solution = self._coefficient_system_product(
            solution,
            weights,
            prior_diagonal,
        )
        cg_residual = rhs - system_solution
        rhs_norm = float(np.linalg.norm(rhs))
        residual_norm = float(np.linalg.norm(cg_residual))
        denominator = max(rhs_norm, np.finfo(self.dtype).tiny)
        relative_residual = residual_norm / denominator
        if relative_residual <= self.coefficient_tolerance:
            return solution.reshape(self.K, 1), 0, relative_residual, None

        if self._basis_norm_squared is None:
            self._basis_norm_squared = np.einsum(
                "ij,ij->j",
                self.U_flat,
                self.U_flat,
            )
        preconditioner = (
            float(np.mean(weights)) * self._basis_norm_squared
            + prior_diagonal
        )
        np.maximum(preconditioner, np.finfo(self.dtype).tiny, out=preconditioner)
        preconditioned = cg_residual / preconditioner
        direction = preconditioned.copy()
        residual_product = float(np.dot(cg_residual, preconditioned))
        max_iterations = (
            self.K
            if self.coefficient_max_iterations is None
            else int(self.coefficient_max_iterations)
        )

        for iteration in range(1, max_iterations + 1):
            system_direction = self._coefficient_system_product(
                direction,
                weights,
                prior_diagonal,
            )
            curvature = float(np.dot(direction, system_direction))
            if not np.isfinite(curvature) or curvature <= 0:
                return None, iteration - 1, relative_residual, "breakdown"

            step = residual_product / curvature
            solution += step * direction
            cg_residual -= step * system_direction
            residual_norm = float(np.linalg.norm(cg_residual))
            relative_residual = residual_norm / denominator
            if not np.isfinite(relative_residual):
                return None, iteration, relative_residual, "non-finite residual"
            if relative_residual <= self.coefficient_tolerance:
                return (
                    solution.reshape(self.K, 1),
                    iteration,
                    relative_residual,
                    None,
                )

            preconditioned = cg_residual / preconditioner
            next_product = float(np.dot(cg_residual, preconditioned))
            if (
                not np.isfinite(next_product)
                or next_product <= 0
                or residual_product <= 0
            ):
                return None, iteration, relative_residual, "breakdown"
            direction *= next_product / residual_product
            direction += preconditioned
            residual_product = next_product

        return None, max_iterations, relative_residual, "iteration limit"

    def update_transform(self) -> None:
        tiny = np.finfo(self.dtype).tiny
        self.P1_ext.reshape(self.M, self.D)[:] = self.P1[:, None]
        P1_safe = np.maximum(self.P1, tiny)[:,None]
        xbar = self.PX / P1_safe
        ix = self._inverse_similarity(xbar)
        E = (ix - self.Y).reshape(self.MD)
        s_for_prior = (self.s if self.with_scale else 1.0)
        gamma = self.lambda_reg * self.sigma2 / (s_for_prior**2)
        prior_diagonal = gamma * self.invL
        diagnostics = self.coefficient_solver_diagnostics
        diagnostics["solve_count"] += 1
        diagnostics["last_fallback_reason"] = None
        selected_solver = self._selected_coefficient_solver()
        if selected_solver == "cg":
            diagnostics["cg_solve_count"] += 1
            b_mle, iterations, relative_residual, failure = (
                self._solve_coefficients_cg(E, prior_diagonal)
            )
            diagnostics["last_iterations"] = iterations
            diagnostics["last_relative_residual"] = relative_residual
            if failure is None:
                diagnostics["last_method"] = "cg"
            else:
                diagnostics["fallback_count"] += 1
                diagnostics["last_method"] = "cholesky"
                diagnostics["last_fallback_reason"] = failure
                b_mle = self._solve_coefficients_cholesky(
                    E,
                    prior_diagonal,
                )
        else:
            diagnostics["last_method"] = "cholesky"
            diagnostics["last_iterations"] = 0
            diagnostics["last_relative_residual"] = None
            b_mle = self._solve_coefficients_cholesky(E, prior_diagonal)
        self.b = b_mle
        self._deformation[:] = self.U_flat.dot(self.b).reshape(self.M, self.D)
        Yb = self.Y + self._deformation
        if self.optimize_similarity:
            R, s, t = self._weighted_similarity_update(Yb, xbar=xbar)
            self.R = R; self.s = float(s if self.with_scale else 1.0); self.t = t
            TY = self._apply_similarity(Yb)
        else:
            TY = Yb
        self.TY = TY; self.TY_world = self._denormalize(TY)
        self.b_diff = float(np.mean(np.abs(self.b - self.prev_b))); self.prev_b[:] = self.b

    def update_variance(self) -> None:
        old = self.sigma2; tiny = np.finfo(self.dtype).tiny
        if getattr(self, "Np", 0.0) <= tiny: self.sigma2 = max(old, tiny)
        else:
            xPx = np.dot(self.Pt1, self.X_sq)
            yPy = np.dot(self.P1, np.sum(self.TY*self.TY, axis=1))
            trPXY = np.sum(self.TY * self.PX); self.sigma2 = (xPx - 2*trPXY + yPy) / (self.Np*self.D)
        if self.sigma2 < self._variance_floor: self.sigma2 = self._variance_floor
        self.r = np.sqrt(self.sigma2)
        self.sigma_diff = abs(self.sigma2 - old)
        self.diff = max(self.sigma_diff / (self.sigma2 + 1e-8), getattr(self, "b_diff", 0.0))

    def _world_similarity_parameters(self) -> Tuple[np.ndarray, float, np.ndarray]:
        if not self.normalize:
            return self.R, self.s, self.t
        c = self.target_centroid
        t_world = -self.s * (c @ self.R.T) + self.target_scale * self.t + c
        return self.R, self.s, np.asarray(t_world, dtype=self.dtype).reshape(1, self.D)

    def get_registration_parameters(self) -> Dict[str, Any]:
        R_world, s_world, t_world = self._world_similarity_parameters()
        return {
            "U_flat": self.U_flat,
            "b": self.b,
            # Backward-compatible aliases. With normalize=True these are internal
            # normalized-space similarity parameters.
            "R": self.R,
            "s": self.s,
            "t": self.t,
            "R_norm": self.R,
            "s_norm": self.s,
            "t_norm": self.t,
            "R_world": R_world,
            "s_world": s_world,
            "t_world": t_world,
        }

    def transformed_points(self, denormalize: bool = True) -> np.ndarray:
        Yb = self.Y + self._deformation
        pts = self._apply_similarity(Yb)
        return self._denormalize(pts) if (denormalize and self.normalize) else pts

    def transform_point_cloud(self, Y: Optional[np.ndarray] = None) -> np.ndarray:
        delta = self._deformation
        base = self.Y if Y is None else np.asarray(Y, dtype=self.dtype)
        if base.shape != (self.M, self.D): raise ValueError("Y must be of shape (M, D).")
        if self.normalize and Y is not None: base = (base - self.target_centroid) / self.target_scale
        out = self._apply_similarity(base + delta)
        if Y is None:
            self.TY = out; self.TY_world = self._denormalize(out); return self.TY_world
        return self._denormalize(out) 
