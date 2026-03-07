from __future__ import division
import numpy as np
import numbers
from warnings import warn


def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).

    Attributes
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (_, D) = X.shape
    muX = np.mean(X, axis=0)
    muY = np.mean(Y, axis=0)
    Xc = X - muX
    Yc = Y - muY
    sigma2 = (
        np.mean(np.sum(Xc * Xc, axis=1))
        + np.mean(np.sum(Yc * Yc, axis=1))
        + np.sum((muX - muY) ** 2)
    ) / D
    return max(float(sigma2), np.finfo(float).tiny)


class EMRegistration(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None,
                 w=None, dense_block_size=None, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        if dense_block_size is not None and (
            not isinstance(dense_block_size, numbers.Number) or dense_block_size <= 0
        ):
            raise ValueError(
                "Expected a positive integer for dense_block_size instead got: {}".format(dense_block_size)
            )
        elif isinstance(dense_block_size, numbers.Number) and not isinstance(dense_block_size, int):
            warn("Received a non-integer value for dense_block_size: {}. Casting to integer.".format(dense_block_size))
            dense_block_size = int(dense_block_size)

        # Use floating point internally for numerically stable EM updates.
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.TY = self.Y.copy()
        self.sigma2 = initialize_sigma2(self.X, self.Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = None
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0
        self.X_sq = np.sum(self.X * self.X, axis=1)
        self._tiny = np.finfo(float).tiny
        self.dense_block_size = dense_block_size

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.

        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.
        
        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.
        
        registration_parameters:
            Returned params dependent on registration method used. 
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        self._compute_dense_posterior_stats(store_p=True)

    def _get_dense_block_size(self):
        if self.dense_block_size is not None:
            return max(1, min(int(self.dense_block_size), self.N))

        # Target roughly 128 MiB across the main dense work buffers.
        target_bytes = 128 * 1024 * 1024
        bytes_per_column = max(self.M, 1) * np.dtype(float).itemsize * 3
        auto_block = max(1, target_bytes // max(bytes_per_column, 1))
        return min(self.N, max(128, int(auto_block)))

    def _compute_dense_posterior_stats(self, store_p=True, TY=None, block_size=None):
        TY_arr = self.TY if TY is None else np.asarray(TY, dtype=float)
        ty_sq = np.sum(TY_arr * TY_arr, axis=1)[:, None]
        block_size = self._get_dense_block_size() if block_size is None else max(1, min(int(block_size), self.N))
        c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1.0 - self.w) * self.M / self.N

        P = np.empty((self.M, self.N), dtype=float) if store_p else None
        Pt1 = np.zeros((self.N,), dtype=float)
        P1 = np.zeros((self.M,), dtype=float)
        PX = np.zeros((self.M, self.D), dtype=float)

        for start in range(0, self.N, block_size):
            stop = min(start + block_size, self.N)
            X_block = self.X[start:stop]
            weights = TY_arr @ X_block.T
            weights *= -2.0
            weights += ty_sq
            weights += self.X_sq[start:stop][None, :]
            np.maximum(weights, 0.0, out=weights)
            weights *= -1.0 / (2.0 * self.sigma2)
            np.exp(weights, out=weights)

            den = np.sum(weights, axis=0, keepdims=True)
            den += c
            np.clip(den, self._tiny, None, out=den)
            weights /= den

            if store_p:
                P[:, start:stop] = weights
            Pt1[start:stop] = np.sum(weights, axis=0)
            P1 += np.sum(weights, axis=1)
            PX += weights @ X_block

        self.P = P
        self.Pt1 = Pt1
        self.P1 = P1
        self.Np = float(P1.sum())
        self.PX = PX

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()
