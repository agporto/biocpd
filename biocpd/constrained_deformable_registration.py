from builtins import super
import numpy as np
import numbers
from .deformable_registration import DeformableRegistration


class ConstrainedDeformableRegistration(DeformableRegistration):
    """
    Constrained deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    e_alpha: float (positive)
        Reliability of correspondence priors. Between 1e-8 (very reliable) and 1 (very unreliable)
    
    source_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the source array
    
    target_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the target array
    
    """

    def __init__(self, e_alpha = None, source_id = None, target_id= None, use_kdtree=True, k=10, *args, **kwargs):
        super().__init__(use_kdtree=use_kdtree, k=k, *args, **kwargs)
        if e_alpha is not None and (not isinstance(e_alpha, numbers.Number) or e_alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter e_alpha. Instead got: {}".format(e_alpha))
        
        if type(source_id) is not np.ndarray or source_id.ndim != 1:
            raise ValueError(
                "The source ids (source_id) must be a 1D numpy array of ints.")
        
        if type(target_id) is not np.ndarray or target_id.ndim != 1:
            raise ValueError(
                "The target ids (target_id) must be a 1D numpy array of ints.")

        self.e_alpha = 1e-8 if e_alpha is None else e_alpha
        self.source_id = source_id
        self.target_id = target_id
        self.P_tilde = np.zeros((self.M, self.N))
        self.P_tilde[self.source_id, self.target_id] = 1
        self.P1_tilde = np.sum(self.P_tilde, axis=1)
        self.PX_tilde = np.dot(self.P_tilde, self.X)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation with correspondence priors.
        Avoids forming large diagonal matrices for performance.
        """
        if self.low_rank is False:
            # A = (diag(P1) + sigma2/e_alpha * diag(P1_tilde)) G + alpha*sigma2*I
            wv = self.P1 + (self.sigma2 * (1.0 / self.e_alpha)) * self.P1_tilde  # (M,)
            A = (wv[:, None] * self.G) + (self.alpha * self.sigma2) * np.eye(self.M)
            # B = PX - diag(P1)Y + sigma2/e_alpha * (PX_tilde - diag(P1_tilde)Y)
            B = self.PX - (self.P1[:, None] * self.Y) + (self.sigma2 * (1.0 / self.e_alpha)) * (self.PX_tilde - (self.P1_tilde[:, None] * self.Y))
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Vector weights instead of explicit diagonal
            dP_vec = self.P1 + (self.sigma2 * (1.0 / self.e_alpha)) * self.P1_tilde  # (M,)
            dPQ = (dP_vec[:, None] * self.Q)  # (M, K)
            F = self.PX - (self.P1[:, None] * self.Y) + (self.sigma2 * (1.0 / self.e_alpha)) * (self.PX_tilde - (self.P1_tilde[:, None] * self.Y))

            # Solve via Woodbury
            M_small = (self.alpha * self.sigma2 * self.inv_S) + (self.Q.T @ dPQ)
            rhs = (self.Q.T @ F)
            inner = np.linalg.solve(M_small, rhs)
            self.W = (1.0 / (self.alpha * self.sigma2)) * (F - dPQ @ inner)