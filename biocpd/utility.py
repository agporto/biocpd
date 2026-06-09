import numpy as np


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    eigvals = np.linalg.eigvals(R)
    return np.all(eigvals >= 0)


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    # Compute pairwise squared distances via the identity
    #   ||x - y||^2 = ||x||^2 - 2 x.y + ||y||^2
    # using a single matrix multiply. This avoids the
    # (len(X), len(Y), D) broadcast temporary that
    # X[:, None, :] - Y[None, :, :] would materialize, dropping peak memory
    # from ~O(len(X)*len(Y)*D) to ~O(len(X)*len(Y)). Unlike
    # scipy.spatial.distance.cdist, which always returns float64, this keeps
    # the computation in the input dtype so float32 pipelines stay float32.
    out_dtype = np.result_type(X, Y, np.float32)
    X = np.ascontiguousarray(X, dtype=out_dtype)
    Y = np.ascontiguousarray(Y, dtype=out_dtype)
    x_sq = np.einsum("ij,ij->i", X, X)
    y_sq = np.einsum("ij,ij->i", Y, Y)
    sq_dist = x_sq[:, None] + y_sq[None, :]
    sq_dist -= 2.0 * (X @ Y.T)
    # Clamp tiny negative values from floating-point cancellation.
    np.maximum(sq_dist, 0.0, out=sq_dist)
    sq_dist *= -1.0 / (2.0 * beta**2)
    np.exp(sq_dist, out=sq_dist)
    return sq_dist





