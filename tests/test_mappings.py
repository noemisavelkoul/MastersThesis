import numpy as np

from src.mappings import a_to_p, p_to_a


def test_a_to_p_and_back_round_trip() -> None:
    a_matrix = np.array([[0.2, -0.1], [0.3, 0.05]])
    p_matrix = a_to_p(a_matrix)
    recovered = p_to_a(p_matrix)
    np.testing.assert_allclose(recovered, a_matrix, atol=1e-8)

