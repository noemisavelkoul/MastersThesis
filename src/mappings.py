import numpy as np

from src.linalg import sqrtm


def a_to_p(a_matrix: np.ndarray) -> np.ndarray:
    m = a_matrix.shape[1]
    c_matrix = np.eye(m) + a_matrix @ a_matrix.T
    b_matrix = sqrtm(c_matrix)
    return np.linalg.solve(b_matrix, a_matrix)


def p_to_a(p_matrix: np.ndarray) -> np.ndarray:
    m = p_matrix.shape[1]
    identity = np.eye(m)
    return sqrtm(np.linalg.solve(identity - p_matrix @ p_matrix.T, identity)) @ p_matrix


def a_stack_to_p_stack(a_matrices: np.ndarray) -> np.ndarray:
    return np.stack([a_to_p(a_matrices[:, :, lag]) for lag in range(a_matrices.shape[2])], axis=2)


def p_to_phi(sigma: np.ndarray, partial_autocorrelations: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    p = len(partial_autocorrelations)
    m = sigma.shape[0]

    sigma_splusone = sigma
    sigma_s = sigma_splusone

    for lag in range(p - 1, -1, -1):
        p_lag = partial_autocorrelations[lag]
        sigma_splusone = sigma_s
        b_inv = sqrtm(np.eye(m) - p_lag @ p_lag.T)
        middle = sqrtm(b_inv @ sigma_splusone @ b_inv)
        s_s = np.linalg.solve(b_inv, middle) @ np.linalg.inv(b_inv)
        sigma_s = s_s @ s_s.T

    gamma: list[np.ndarray] = [sigma_s]
    sigma_0 = gamma[0]
    sigma_0_star = gamma[0]
    s_0 = sqrtm(sigma_0)
    s_0_star = s_0

    phi_splusone = [s_0 @ partial_autocorrelations[0] @ np.linalg.inv(s_0_star)]
    phi_splusone_star = [s_0_star @ partial_autocorrelations[0].T @ np.linalg.inv(s_0)]

    sigma_splusone = sigma_0 - phi_splusone[0] @ sigma_0_star @ phi_splusone[0].T
    sigma_splusone_star = sigma_0_star - phi_splusone_star[0] @ sigma_0 @ phi_splusone_star[0].T
    s_splusone = sqrtm(sigma_splusone)
    s_splusone_star = sqrtm(sigma_splusone_star)
    gamma.append((phi_splusone[0] @ sigma_0_star).T)

    for splusone in range(2, p + 1):
        phi_s = phi_splusone.copy()
        phi_s_star = phi_splusone_star.copy()
        sigma_s = sigma_splusone
        sigma_s_star = sigma_splusone_star
        s_s = s_splusone
        s_s_star = s_splusone_star

        phi_splusone = [np.zeros((m, m)) for _ in range(splusone)]
        phi_splusone_star = [np.zeros((m, m)) for _ in range(splusone)]
        phi_splusone[-1] = s_s @ partial_autocorrelations[splusone - 1] @ np.linalg.inv(s_s_star)
        phi_splusone_star[-1] = (
            s_s_star @ partial_autocorrelations[splusone - 1].T @ np.linalg.inv(s_s)
        )

        for i in range(splusone - 1):
            phi_splusone[i] = phi_s[i] - phi_splusone[-1] @ phi_s_star[splusone - i - 2]
            phi_splusone_star[i] = (
                phi_s_star[i] - phi_splusone_star[-1] @ phi_s[splusone - i - 2]
            )

        sigma_splusone = sigma_s - phi_splusone[-1] @ sigma_s_star @ phi_splusone[-1].T
        sigma_splusone_star = (
            sigma_s_star - phi_splusone_star[-1] @ sigma_s @ phi_splusone_star[-1].T
        )
        s_splusone = sqrtm(sigma_splusone)
        s_splusone_star = sqrtm(sigma_splusone_star)

        gamma_temp = phi_splusone[-1] @ sigma_s_star
        for i in range(splusone - 1):
            gamma_temp = gamma_temp + phi_s[i] @ gamma[splusone - i - 1].T
        gamma.append(gamma_temp.T)

    return phi_splusone, gamma

