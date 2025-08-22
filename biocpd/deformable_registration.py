from builtins import super
import numpy as np
import numbers
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from .emregistration import EMRegistration
from .utility import gaussian_kernel

from sklearn.utils.extmath import randomized_svd

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
    use_kdtree: bool
        If True, accelerates the E-step by using a k-d tree for nearest neighbor
        searches.
    radius_mode: bool
        If True, in sparse E-step ignore neighbors beyond a numerically safe radius.
    w: float
        Outlier weight (0 <= w < 1) forwarded to EM base.
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
            Number of eigenmodes to use for low-rank approximation. Defaults to 100.
        use_kdtree: bool, optional
            Whether to use a k-d tree for acceleration. Defaults to True.
        k: int, optional
            Number of nearest neighbors to query in the k-d tree. Defaults to 10.
        radius_mode: bool, optional
            If True, mask neighbors beyond sigma-derived radius in sparse E-step.
        w: float, optional
            Outlier weight in [0,1). Forwarded to EM base.
        """
        super().__init__(w=w, *args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(f"alpha must be a positive number, but got {alpha}")
        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(f"beta must be a positive number, but got {beta}")

        self.alpha = 2.0 if alpha is None else alpha
        self.beta = 2.0 if beta is None else beta
        self.low_rank = low_rank
        self.num_eig = num_eig
        self.W = np.zeros((self.M, self.D))

        # Pre-compute Gaussian kernel or its low-rank approximation.
        if self.low_rank:
            G_full = gaussian_kernel(self.Y, self.beta)
            U, S, _ = randomized_svd(G_full, n_components=self.num_eig, n_iter=3)
            self.Q = U         # M x num_eig
            self.S = np.diag(S) # num_eig x num_eig
            self.inv_S = np.diag(1. / S)
        else:
            self.G = gaussian_kernel(self.Y, self.beta)

        # K-D tree setup for accelerated E-step.
        self.use_kdtree = use_kdtree
        self.radius_mode = bool(radius_mode)
        if self.use_kdtree:
            self.k = k
            self.kdtree = cKDTree(self.X)

    def expectation(self):
        """Compute E-step; optionally k-NN sparse with radius gating."""
        if self.use_kdtree:
            distances, indices = self.kdtree.query(self.TY, k=self.k)
            distances = np.clip(distances, np.finfo(self.X.dtype).eps, None)

            # mask invalid neighbors
            mask = np.isfinite(distances)
            if distances.ndim == 1:
                distances = distances[:, None]; indices = indices[:, None]; mask = mask[:, None]
            # Optional radius gating
            if self.radius_mode:
                rad = float(np.sqrt(-2.0 * self.sigma2 * np.log(np.finfo(self.X.dtype).eps)))
                mask = mask & (distances <= rad)

            if not mask.any():
                # fall back to dense if no valid sparse candidates
                return super().expectation()

            P_vals = np.exp(-(np.minimum(distances, np.sqrt(-2.0 * self.sigma2 * np.log(np.finfo(self.X.dtype).eps))) ** 2) / (2 * self.sigma2))

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
            den_col = np.array(P_sparse.sum(axis=0)).ravel() + c_term
            den_col = np.clip(den_col, np.finfo(self.X.dtype).eps, None)
            inv_den_col = 1.0 / den_col

            self.P = P_sparse.multiply(inv_den_col[np.newaxis, :])
            self.Pt1 = np.array(self.P.sum(axis=0)).ravel()
            self.P1 = np.array(self.P.sum(axis=1)).ravel()
            self.Np = self.P1.sum()
            self.PX = self.P @ self.X
        else:
            super().expectation()

    def update_transform(self):
        """
        Updates the transformation parameters (M-step).

        This method solves for the deformation field weights `W`. If `low_rank`
        is enabled, it uses the Woodbury matrix identity to solve the linear system
        efficiently. Otherwise, it solves the full-rank system directly.
        """
        if not self.low_rank:
            A = (self.P1[:, None] * self.G) + self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - (self.P1[:, None] * self.Y)
            self.W = np.linalg.solve(A, B)
        else:
            # Efficiently solve for W using the Woodbury matrix identity.
            # The system is (diag(P1)QSQ' + lambda*I)W = F, where lambda = alpha*sigma^2.
            # The solution is W = (1/lambda) * (F - DQ(lambda*S^-1 + Q'DQ)^-1 * Q'F).
            # The following code implements this solution.
            lambda_val = self.alpha * self.sigma2
            F = self.PX - (self.P1[:, np.newaxis] * self.Y)
            
            # Mmat = (lambda*S^-1 + Q'DQ)
            mmat_term = self.Q.T @ (self.P1[:, np.newaxis] * self.Q)
            Mmat = lambda_val * self.inv_S + mmat_term

            # sol = (Mmat)^-1 * Q'F
            sol = np.linalg.solve(Mmat, self.Q.T @ F)
            
            # W = (1/lambda) * (F - DQ * sol)
            self.W = (F - (self.P1[:, np.newaxis] * self.Q) @ sol) / lambda_val


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
            return Y + G_new @ self.W
        else:
            # Transform the original source point cloud.
            if self.low_rank:
                # Use G = QSQ' approximation
                self.TY = self.Y + self.Q @ self.S @ (self.Q.T @ self.W)
            else:
                self.TY = self.Y + self.G @ self.W
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

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10.
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