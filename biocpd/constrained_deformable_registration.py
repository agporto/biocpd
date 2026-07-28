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
        if self.optimize_similarity:
            raise ValueError(
                "optimize_similarity is currently supported only by "
                "DeformableRegistration."
            )
        if e_alpha is not None and (not isinstance(e_alpha, numbers.Number) or e_alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter e_alpha. Instead got: {}".format(e_alpha))
        
        if type(source_id) is not np.ndarray or source_id.ndim != 1:
            raise ValueError(
                "The source ids (source_id) must be a 1D numpy array of ints.")
        
        if type(target_id) is not np.ndarray or target_id.ndim != 1:
            raise ValueError(
                "The target ids (target_id) must be a 1D numpy array of ints.")
        if not np.issubdtype(source_id.dtype, np.integer):
            raise ValueError("source_id must contain integer indices.")
        if not np.issubdtype(target_id.dtype, np.integer):
            raise ValueError("target_id must contain integer indices.")
        if len(source_id) != len(target_id):
            raise ValueError("source_id and target_id must have the same length.")
        if (
            np.any(source_id < 0)
            or np.any(source_id >= self.M)
            or np.any(target_id < 0)
            or np.any(target_id >= self.N)
        ):
            raise ValueError("constraint indices are out of bounds.")

        self.e_alpha = 1e-8 if e_alpha is None else e_alpha
        self.source_id = source_id
        self.target_id = target_id
        pairs = np.unique(
            np.column_stack((self.source_id, self.target_id)),
            axis=0,
        )
        self._constraint_source_id = pairs[:, 0]
        self._constraint_target_id = pairs[:, 1]
        self._P_tilde = None
        self.P1_tilde = np.bincount(
            self._constraint_source_id,
            minlength=self.M,
        ).astype(self.dtype, copy=False)
        self.PX_tilde = np.zeros((self.M, self.D), dtype=self.dtype)
        np.add.at(
            self.PX_tilde,
            self._constraint_source_id,
            self.X[self._constraint_target_id],
        )

    @property
    def P_tilde(self):
        """Return the historical dense constraint matrix, built on demand."""
        if self._P_tilde is None:
            self._P_tilde = np.zeros((self.M, self.N), dtype=self.dtype)
            self._P_tilde[
                self._constraint_source_id,
                self._constraint_target_id,
            ] = 1
        return self._P_tilde

    @P_tilde.setter
    def P_tilde(self, value):
        """Preserve assignment compatibility for the historical attribute."""
        self._P_tilde = np.asarray(value, dtype=self.dtype)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation with correspondence priors.
        Avoids forming large diagonal matrices for performance.
        """
        if self.low_rank is False:
            # A = (diag(P1) + sigma2/e_alpha * diag(P1_tilde)) G + alpha*sigma2*I
            wv = self.P1 + (self.sigma2 * (1.0 / self.e_alpha)) * self.P1_tilde  # (M,)
            A = (wv[:, None] * self.G) + (self.alpha * self.sigma2) * np.eye(self.M, dtype=self.dtype)
            # B = PX - diag(P1)Y + sigma2/e_alpha * (PX_tilde - diag(P1_tilde)Y)
            B = self.PX - (self.P1[:, None] * self.Y) + (self.sigma2 * (1.0 / self.e_alpha)) * (self.PX_tilde - (self.P1_tilde[:, None] * self.Y))
            self.W = np.linalg.solve(A, B)
            self._low_rank_coefficients = None
            self._low_rank_coefficients_current = False

        elif self.low_rank is True:
            # Vector weights instead of explicit diagonal
            dP_vec = self.P1 + (self.sigma2 * (1.0 / self.e_alpha)) * self.P1_tilde  # (M,)
            F = self.PX - (self.P1[:, None] * self.Y) + (self.sigma2 * (1.0 / self.e_alpha)) * (self.PX_tilde - (self.P1_tilde[:, None] * self.Y))

            self._update_low_rank_transform(
                dP_vec,
                F,
                self.dtype.type(self.alpha * self.sigma2),
            )
