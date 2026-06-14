import numpy as np

from src.truncation import calculate_threshold, compute_optimal_p


def test_threshold_is_positive() -> None:
    assert calculate_threshold(m=3, n_observations=10_000, beta=0.99) > 0


def test_compute_optimal_p_stops_at_first_small_lag() -> None:
    p_matrices = np.zeros((1, 1, 3))
    p_matrices[:, :, 0] = 0.5
    assert compute_optimal_p(p_matrices, m=1, n_observations=100, beta=0.99, p_max=3) == 1

