import numpy as np
from numpy.random import Generator

from src.linalg import companion_matrix, is_positive_definite
from src.mappings import a_to_p, p_to_phi


def simulate_a_matrices(m: int, p: int, rng: Generator) -> list[np.ndarray]:
    return [rng.normal(0.0, 1.0, size=(m, m)) for _ in range(p)]


def check_stationarity(phi: list[np.ndarray]) -> bool:
    eigenvalues = np.linalg.eigvals(companion_matrix(phi))
    return bool(np.all(np.abs(eigenvalues) < 1))


def simulate_var(
    n_observations: int,
    phi: list[np.ndarray],
    sigma: np.ndarray,
    gamma: list[np.ndarray],
    rng: Generator,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    p = len(phi)
    m = sigma.shape[0]

    if mu is None:
        mu = np.zeros(m)
    if not is_positive_definite(sigma):
        raise ValueError("sigma must be positive definite.")
    if not check_stationarity(phi):
        raise ValueError("VAR parameters do not lie in the stationary region.")

    stationary_cov = np.empty((p * m, p * m))
    for i in range(p):
        for j in range(p):
            block = gamma[j - i] if i <= j else gamma[i - j].T
            stationary_cov[i * m : (i + 1) * m, j * m : (j + 1) * m] = block

    y = np.empty((n_observations, m))
    initial_mean = np.tile(mu, p)
    y[:p, :] = rng.multivariate_normal(initial_mean, stationary_cov).reshape(p, m)

    for t in range(p, n_observations):
        mean = mu.copy()
        for lag in range(1, p + 1):
            mean = mean + phi[lag - 1] @ (y[t - lag] - mu)
        y[t] = rng.multivariate_normal(mean, sigma)

    return y


def generate_dataset(
    n_observations: int,
    m: int,
    p: int,
    sigma: np.ndarray,
    mu: np.ndarray,
    rng: Generator,
) -> dict[str, object]:
    a_matrices = simulate_a_matrices(m, p, rng)
    partial_autocorrelations = [a_to_p(a_matrix) for a_matrix in a_matrices]
    phi, gamma = p_to_phi(sigma, partial_autocorrelations)
    y = simulate_var(n_observations, phi, sigma, gamma, rng, mu)

    return {
        "y": y,
        "mu": mu,
        "phi": phi,
        "sigma": sigma,
        "a": a_matrices,
        "p": partial_autocorrelations,
        "gamma": gamma,
    }


def mean_center(y: np.ndarray) -> np.ndarray:
    return y - np.mean(y, axis=0, keepdims=True)

