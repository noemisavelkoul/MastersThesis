import numpy as np
from scipy.stats import norm


def calculate_threshold(m: int, n_observations: int, beta: float) -> float:
    return float(norm.ppf((beta ** (1 / m**2) + 1) / 2) / np.sqrt(n_observations))


def compute_optimal_p(
    p_matrices: np.ndarray,
    m: int,
    n_observations: int,
    beta: float,
    p_max: int,
) -> int:
    threshold = calculate_threshold(m, n_observations, beta)
    optimal_p = 0

    for lag in range(p_max):
        if np.max(np.abs(p_matrices[:, :, lag])) <= threshold:
            break
        optimal_p = lag + 1

    return optimal_p


def effective_order_from_draws(
    p_draws: np.ndarray,
    p_max: int,
    threshold: float,
    use_singular_value: bool = False,
) -> dict[str, np.ndarray]:
    iterations = p_draws.shape[0]
    m = int(np.sqrt(p_draws.shape[1] / p_max))
    nonzero = np.zeros((iterations, p_max), dtype=bool)

    for iteration in range(iterations):
        p_array = p_draws[iteration, :].reshape((p_max, m, m))
        if use_singular_value:
            nonzero[iteration, :] = [
                np.linalg.svd(p_array[lag], compute_uv=False)[0] >= threshold
                for lag in range(p_max)
            ]
        else:
            nonzero[iteration, :] = [
                not np.all(np.abs(p_array[lag]) < threshold) for lag in range(p_max)
            ]

    samples = np.array([np.max(np.flatnonzero(row)) + 1 if row.any() else 0 for row in nonzero])
    pmf = np.bincount(samples, minlength=p_max + 1)[: p_max + 1] / iterations

    return {"nonzero": nonzero, "samples": samples, "pmf": pmf}
