import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import sqrtm as scipy_sqrtm


def as_real_matrix(matrix: ArrayLike, *, atol: float = 1e-10) -> np.ndarray:
    values = np.asarray(matrix)
    if np.iscomplexobj(values):
        if not np.allclose(values.imag, 0.0, atol=atol):
            raise ValueError("Expected a real-valued matrix, got a matrix with complex part.")
        values = values.real
    return np.asarray(values, dtype=float)


def sqrtm(matrix: ArrayLike) -> np.ndarray:
    return as_real_matrix(scipy_sqrtm(np.asarray(matrix, dtype=float)))


def is_positive_definite(matrix: ArrayLike) -> bool:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        return False
    try:
        np.linalg.cholesky(values)
    except np.linalg.LinAlgError:
        return False
    return True


def companion_matrix(phi: list[np.ndarray]) -> np.ndarray:
    if not phi:
        raise ValueError("phi must contain at least one coefficient matrix.")

    p = len(phi)
    m = phi[0].shape[0]
    companion = np.zeros((p * m, p * m))

    for lag, phi_lag in enumerate(phi):
        companion[:m, lag * m : (lag + 1) * m] = phi_lag

    for lag in range(1, p):
        companion[lag * m : (lag + 1) * m, (lag - 1) * m : lag * m] = np.eye(m)

    return companion

