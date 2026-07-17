"""Low-rank Gaussian-kernel factorizations used by deformable CPD."""

import numbers

import numpy as np

from .utility import gaussian_kernel


LOW_RANK_METHODS = ("randomized_svd", "pivoted_cholesky")


def validate_low_rank_options(num_eig, method, tolerance):
    """Validate and normalize low-rank factorization options."""
    if not isinstance(num_eig, numbers.Integral) or isinstance(num_eig, bool):
        raise ValueError("num_eig must be a positive integer.")
    if num_eig <= 0:
        raise ValueError("num_eig must be a positive integer.")
    if method not in LOW_RANK_METHODS:
        raise ValueError(
            "low_rank_method must be one of {}.".format(
                ", ".join(repr(value) for value in LOW_RANK_METHODS)
            )
        )
    if (
        not isinstance(tolerance, numbers.Number)
        or not np.isfinite(tolerance)
        or tolerance < 0
        or tolerance >= 1
    ):
        raise ValueError("low_rank_tolerance must be in [0, 1).")
    return int(num_eig), str(method), float(tolerance)


def _pivoted_cholesky_factor(points, beta, rank, tolerance):
    """Return ``L`` such that the Gaussian kernel is approximated by ``L L.T``.

    Kernel columns are evaluated on demand. The full square kernel is never
    materialized, so memory grows linearly with the point count for fixed rank.
    """
    points = np.ascontiguousarray(points)
    point_count = len(points)
    rank = min(int(rank), point_count)
    factor = np.empty((point_count, rank), dtype=points.dtype)
    squared_norms = np.einsum("ij,ij->i", points, points)
    residual_diagonal = np.ones(point_count, dtype=points.dtype)
    numerical_floor = 10.0 * np.finfo(points.dtype).eps
    stopping_threshold = max(float(tolerance), numerical_floor)
    completed_rank = 0

    for column_index in range(rank):
        pivot = int(np.argmax(residual_diagonal))
        pivot_value = float(residual_diagonal[pivot])
        if pivot_value <= stopping_threshold:
            break

        column = points @ points[pivot]
        column *= -2.0
        column += squared_norms
        column += squared_norms[pivot]
        np.maximum(column, 0.0, out=column)
        column *= -1.0 / (2.0 * beta**2)
        np.exp(column, out=column)

        if column_index:
            column -= factor[:, :column_index] @ factor[pivot, :column_index]
        column /= np.sqrt(pivot_value)
        factor[:, column_index] = column

        residual_diagonal -= column * column
        np.maximum(residual_diagonal, 0.0, out=residual_diagonal)
        residual_diagonal[pivot] = 0.0
        completed_rank += 1

    return factor[:, :completed_rank], residual_diagonal


def _positive_spectrum(basis, eigenvalues):
    """Return a finite positive spectrum while preserving the factor rank."""
    eigenvalues = np.asarray(eigenvalues, dtype=basis.dtype)
    if not np.isfinite(eigenvalues).all():
        raise np.linalg.LinAlgError("low-rank kernel spectrum is not finite")

    scale = max(float(eigenvalues[0]) if eigenvalues.size else 0.0, 1.0)
    floor = np.finfo(basis.dtype).eps * scale
    if not eigenvalues.size or float(np.max(eigenvalues)) <= 0:
        raise np.linalg.LinAlgError("low-rank kernel has no positive eigenvalues")
    eigenvalues = np.maximum(eigenvalues, floor)
    return (
        np.ascontiguousarray(basis, dtype=basis.dtype),
        np.ascontiguousarray(eigenvalues, dtype=basis.dtype),
    )


def gaussian_kernel_eigendecomposition(
    points,
    beta,
    rank,
    method="randomized_svd",
    tolerance=0.0,
):
    """Return ``Q, eigenvalues, diagnostics`` for ``G ~= Q diag(s) Q.T``."""
    points = np.ascontiguousarray(points)
    rank = min(int(rank), len(points))

    if method == "randomized_svd":
        # Keep this import lazy: users of other registration methods should not
        # pay scikit-learn's import cost.
        from sklearn.utils.extmath import randomized_svd

        kernel = gaussian_kernel(points, beta)
        basis, eigenvalues, _ = randomized_svd(
            kernel,
            n_components=rank,
            n_iter=3,
        )
        basis, eigenvalues = _positive_spectrum(basis, eigenvalues)
        diagnostics = {
            "method": method,
            "requested_rank": rank,
            "rank": len(eigenvalues),
            "max_residual_diagonal": None,
        }
        return basis, eigenvalues, diagnostics

    if method == "pivoted_cholesky":
        factor, residual_diagonal = _pivoted_cholesky_factor(
            points,
            beta,
            rank,
            tolerance,
        )
        basis, singular_values, _ = np.linalg.svd(
            factor,
            full_matrices=False,
        )
        eigenvalues = singular_values * singular_values
        basis, eigenvalues = _positive_spectrum(basis, eigenvalues)
        diagnostics = {
            "method": method,
            "requested_rank": rank,
            "rank": len(eigenvalues),
            "max_residual_diagonal": float(np.max(residual_diagonal)),
        }
        return basis, eigenvalues, diagnostics

    raise ValueError("unsupported low-rank method: {}".format(method))
