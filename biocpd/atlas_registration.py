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
        Mean shape as flattened (M*D,) or (M, D). If provided, replaces `Y` as the
        initial shape in the model's space.
    U : Optional[np.ndarray]
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
    optimize_similarity : bool
        If True, optimize similarity (R, s, t) each iteration via weighted Procrustes.
    with_scale : bool
        If True, include uniform scale in similarity.
    radius_mode : bool
        If True, in sparse E-step ignore neighbors with distance greater than a
        numerically safe radius tied to current sigma2.
    w : float
        Outlier weight (0 <= w < 1). Forwarded to EM base.
    **kwargs : Any
        Forwarded to EM base (e.g., max_iterations, tolerance).
    """
    @staticmethod
    def _normalize_point_cloud(pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        pc = np.asarray(pc, float); c = np.mean(pc, axis=0)
        centered = pc - c; s = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        return (centered / (s if s>=np.finfo(float).eps else 1.0), c, s if s>=np.finfo(float).eps else 1.0)

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 *,
                 mean_shape: Optional[np.ndarray] = None,
                 U: Optional[np.ndarray] = None,
                 eigenvalues: Optional[np.ndarray] = None,
                 lambda_reg: float = 0.1,
                 normalize: bool = False,
                 use_kdtree: bool = True,
                 k: int = 10,
                 kdtree_radius_scale: float = 10.0,
                 optimize_similarity: bool = True,
                 with_scale: bool = True,
                 radius_mode: bool = False,
                 w: float = 0.0,
                 **kwargs: Any) -> None:
        if lambda_reg is None or eigenvalues is None: raise ValueError("lambda_reg and eigenvalues must be provided.")
        self.normalize = normalize
        X_raw = np.asarray(X, float); Y_raw = np.asarray(Y, float)
        if normalize:
            Xn, c, s = self._normalize_point_cloud(X_raw); Yn = (Y_raw - c) / s
            self.target_centroid, self.target_scale = c, s; X, Y = Xn, Yn
            if mean_shape is not None:
                ms = (np.asarray(mean_shape, float) - c) / s; mean_shape = ms.reshape(-1) if ms.ndim == 2 else ms
            if U is not None: U = np.asarray(U, float) / s
        else:
            X, Y = X_raw, Y_raw; self.target_centroid = np.zeros(Y_raw.shape[1]); self.target_scale = 1.0
        super().__init__(X=X, Y=Y, w=w, **kwargs)

        self.use_kdtree = use_kdtree; self._use_sparse = False
        self.N, self.D = self.X.shape; self.M = self.Y.shape[0]
        self.lambda_reg = float(lambda_reg); self.optimize_similarity = bool(optimize_similarity); self.with_scale = bool(with_scale)
        self.radius_mode = bool(radius_mode)

        ev = np.asarray(eigenvalues, float); 
        if ev.ndim != 1: raise ValueError("eigenvalues has an invalid shape.")
        U_arr = np.asarray(U, float); self.K = ev.shape[0]; self.MD = self.M * self.D
        if U_arr.ndim == 3 and U_arr.shape == (self.M, self.D, self.K): self.U_flat = U_arr.reshape(self.MD, self.K)
        elif U_arr.ndim == 2 and U_arr.shape == (self.MD, self.K): self.U_flat = U_arr.copy()
        else: raise ValueError("U has an invalid shape.")

        if mean_shape is not None:
            ms = np.asarray(mean_shape, float)
            if ms.ndim == 1 and ms.size == self.MD: self.mean_shape = ms.reshape(self.M, self.D)
            elif ms.ndim == 2 and ms.shape == (self.M, self.D): self.mean_shape = ms.copy()
            else: raise ValueError("mean_shape has an invalid shape.")
            self.Y = self.mean_shape.copy()
        else: self.mean_shape = None

        self.L = (ev/(self.target_scale*self.target_scale)).copy() if self.normalize else ev.copy()
        if self.L.shape != (self.K,): raise ValueError("eigenvalues has an invalid shape.")
        self.L = np.asarray(self.L, dtype=float)
        floor = max(1e-12, np.finfo(self.L.dtype).eps)
        self.L = np.clip(self.L, floor, None)
        self.invL = 1.0 / self.L

        dx = self.X.max(0) - self.X.min(0)
        self.kdtree_radius_threshold = float(np.linalg.norm(dx)) / float(kdtree_radius_scale)
        self.k = max(1, min(int(k), self.N))
        if use_kdtree: self.kdtree = cKDTree(self.X)

        self.r = np.sqrt(self.sigma2)
        self.R = np.eye(self.D); self.s = 1.0; self.t = np.zeros((1, self.D))
        self.TY = self.Y.copy(); self.TY_world = self._denormalize(self.TY)
        self.Y_flat = self.Y.ravel()

        self.P1_ext = np.empty(self.MD); self.WU = np.empty((self.MD, self.K))
        self.b = np.zeros((self.K, 1)); self.prev_b = self.b.copy()

    def _denormalize(self, pts: np.ndarray) -> np.ndarray:
        return pts*self.target_scale + self.target_centroid if self.normalize else pts
    def _apply_similarity(self, Z: np.ndarray) -> np.ndarray: return (self.s * (Z @ self.R.T)) + self.t
    def _inverse_similarity(self, Z: np.ndarray) -> np.ndarray:
        s = self.s if self.with_scale else 1.0
        s = s if s>np.finfo(float).tiny else 1.0
        return (Z - self.t) @ self.R / s

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

    def expectation(self) -> None:
        if self.use_kdtree and not self._use_sparse and self.sigma2 < self.kdtree_radius_threshold ** 2:
            self._use_sparse = True

        if self._use_sparse:
            valid_rows = np.all(np.isfinite(self.TY), axis=1)
            TYq = self.TY[valid_rows] if not valid_rows.all() else self.TY

            if TYq.size == 0:
                self._use_sparse = False
                return self.expectation()

            d, idx = self.kdtree.query(TYq, k=self.k)
            if d.ndim == 1:
                d = d[:, None]; idx = idx[:, None]
            else:
                d   = np.asarray(d,   dtype=float,   order="C")
                idx = np.asarray(idx, dtype=np.int64, order="C")

            mask = np.isfinite(d) & (idx >= 0) & (idx < self.N)
            # Optional radius gating
            if self.radius_mode:
                rad = float(np.sqrt(-2.0 * self.sigma2 * np.log(np.finfo(self.X.dtype).eps)))
                mask &= (d <= rad)
            if not mask.any():
                self._use_sparse = False
                return self.expectation()

            Pv = np.exp(-(np.minimum(d, np.sqrt(-2.0 * self.sigma2 * np.log(np.finfo(self.X.dtype).eps))) ** 2) / (2.0 * self.sigma2))

            base_rows = (np.where(valid_rows)[0] if not valid_rows.all() else np.arange(self.M))
            rows_all  = np.repeat(base_rows, self.k)
            rows = rows_all[mask.ravel()]
            cols = idx.ravel()[mask.ravel()]
            vals = Pv.ravel()[mask.ravel()]

            if cols.size:
                mx = int(cols.max()); mn = int(cols.min())
                if not (0 <= mn and mx < self.N):
                    keep = (cols >= 0) & (cols < self.N)
                    rows, cols, vals = rows[keep], cols[keep], vals[keep]

            Psp = csr_matrix((vals, (rows, cols)), shape=(self.M, self.N))

            c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1 - self.w) * self.M / self.N
            den = np.asarray(Psp.sum(axis=0)).ravel() + c
            den = np.maximum(den, np.finfo(float).tiny)

            Psp = Psp.tocsr()
            Psp.data *= (1.0 / den)[Psp.indices]
            self.P = Psp

            self.Pt1 = np.asarray(self.P.sum(axis=0)).ravel()
            self.P1  = np.asarray(self.P.sum(axis=1)).ravel()
            self.Np  = float(self.P1.sum())
            self.PX  = self.P @ self.X
            return

        diff2 = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)
        P = np.exp(-diff2 / (2 * self.sigma2))
        c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1 - self.w) * self.M / self.N
        den = np.sum(P, axis=0, keepdims=True) + c
        den = np.clip(den, np.finfo(float).tiny, None)
        self.P   = P / den
        self.Pt1 = self.P.sum(axis=0)
        self.P1  = self.P.sum(axis=1)
        self.Np  = float(self.P1.sum())
        self.PX  = self.P @ self.X

    def _weighted_similarity_update(self, Yb: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        tiny = np.finfo(float).tiny
        w = self.P1; wsum = max(self.Np, tiny)
        xbar = self.PX / np.maximum(w[:,None], tiny)
        mu_x = (w[:,None]*xbar).sum(axis=0, keepdims=True)/wsum
        mu_y = (w[:,None]*Yb ).sum(axis=0, keepdims=True)/wsum
        Xc = xbar - mu_x; Yc = Yb - mu_y
        C = (Yc.T * w).dot(Xc)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        M = np.eye(self.D); M[-1,-1] = np.sign(np.linalg.det(U@Vt))
        R = U @ M @ Vt
        if self.with_scale:
            num = np.trace(R.T @ C)
            den = float((w * (Yc*Yc).sum(axis=1)).sum()); s = float(num / max(den, tiny))
        else:
            s = 1.0
        t = mu_x - s*(mu_y @ R.T)
        return R, s, t

    def update_transform(self) -> None:
        tiny = np.finfo(float).tiny
        self.P1_ext[:] = np.repeat(self.P1, self.D)
        P1_safe = np.maximum(self.P1, tiny)[:,None]
        xbar = self.PX / P1_safe
        ix = self._inverse_similarity(xbar)
        E = (ix - self.Y).reshape(self.MD)
        self.WU[:] = self.U_flat * self.P1_ext[:,None]
        A = self.U_flat.T.dot(self.WU)
        s_for_prior = (self.s if self.with_scale else 1.0)
        gamma = self.lambda_reg * self.sigma2 / (s_for_prior**2)
        A[np.diag_indices(self.K)] += gamma * self.invL
        rhs = self.U_flat.T.dot(self.P1_ext * E).reshape(self.K,1)
        try:
            c, low = cho_factor(A, overwrite_a=True, check_finite=False)
            b_mle = cho_solve((c, low), rhs, overwrite_b=True, check_finite=False)
        except np.linalg.LinAlgError:
            A[np.diag_indices(self.K)] += 1e-8
            c, low = cho_factor(A, overwrite_a=False, check_finite=False)
            b_mle = cho_solve((c, low), rhs, overwrite_b=False, check_finite=False)
        self.b = b_mle
        Yb = self.Y + self.U_flat.dot(self.b).reshape(self.M, self.D)
        if self.optimize_similarity:
            R, s, t = self._weighted_similarity_update(Yb)
            self.R = R; self.s = float(s if self.with_scale else 1.0); self.t = t
            TY = self._apply_similarity(Yb)
        else:
            TY = Yb
        self.TY = TY; self.TY_world = self._denormalize(TY)
        self.b_diff = float(np.mean(np.abs(self.b - self.prev_b))); self.prev_b[:] = self.b

    def update_variance(self) -> None:
        old = self.sigma2; tiny = np.finfo(float).tiny
        if getattr(self, "Np", 0.0) <= tiny: self.sigma2 = max(old, tiny)
        else:
            xPx = np.dot(self.Pt1, np.sum(self.X*self.X, axis=1))
            yPy = np.dot(self.P1, np.sum(self.TY*self.TY, axis=1))
            trPXY = np.sum(self.TY * self.PX); self.sigma2 = (xPx - 2*trPXY + yPy) / (self.Np*self.D)
        bb = self.X.max(0) - self.X.min(0); floor = 1e-6 * float(np.dot(bb, bb)) / self.D
        if self.sigma2 < floor: self.sigma2 = floor
        self.r = np.sqrt(self.sigma2)
        self.sigma_diff = abs(self.sigma2 - old)
        self.diff = max(self.sigma_diff / (self.sigma2 + 1e-8), getattr(self, "b_diff", 0.0))

    def get_registration_parameters(self) -> Dict[str, Any]:
        return {"U_flat": self.U_flat, "b": self.b, "R": self.R, "s": self.s, "t": self.t}

    def transformed_points(self, denormalize: bool = True) -> np.ndarray:
        Yb = self.Y + self.U_flat.dot(self.b).reshape(self.M, self.D)
        pts = self._apply_similarity(Yb)
        return self._denormalize(pts) if (denormalize and self.normalize) else pts

    def transform_point_cloud(self, Y: Optional[np.ndarray] = None) -> np.ndarray:
        delta = self.U_flat.dot(self.b).reshape(self.M, self.D)
        base = self.Y if Y is None else np.asarray(Y, float)
        if base.shape != (self.M, self.D): raise ValueError("Y must be of shape (M, D).")
        if self.normalize and Y is not None: base = (base - self.target_centroid) / self.target_scale
        out = self._apply_similarity(base + delta)
        if Y is None:
            self.TY = out; self.TY_world = self._denormalize(out); return self.TY_world
        return self._denormalize(out) 