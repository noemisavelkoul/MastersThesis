import numpy as np
from numpy.random import Generator

from src.config import MGPHyperparameters, PriorConfig
from src.mappings import a_stack_to_p_stack
from src.truncation import compute_optimal_p


def generate_tau(a1: float, a2: float, p_max: int, rng: Generator) -> np.ndarray:
    gamma_shrinkage = np.concatenate(([rng.gamma(a1, 1.0)], rng.gamma(a2, 1.0, p_max - 1)))
    return np.cumprod(gamma_shrinkage)


def generate_lambda_original(a: float, m: int, p_max: int, rng: Generator, constant: float = 0.0) -> np.ndarray:
    rate = a / 2 + constant
    return rng.gamma(shape=a / 2, scale=1 / rate, size=(m, m, p_max))


def generate_lambda_extended(
    c1: float,
    c2: float,
    a: float,
    m: int,
    p_max: int,
    rng: Generator,
) -> np.ndarray:
    eta = rng.gamma(shape=c1, scale=1 / c2, size=(m, m, p_max))
    return rng.gamma(shape=a / 2, scale=1 / eta)


def generate_a_matrices(lambda_values: np.ndarray, tau: np.ndarray, rng: Generator) -> np.ndarray:
    variance = 1 / (lambda_values * tau[np.newaxis, np.newaxis, :])
    return rng.normal(loc=0.0, scale=np.sqrt(variance))


def simulate_prior_effective_orders(
    config: PriorConfig,
    hyperparameters: MGPHyperparameters,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    orders = np.empty(config.samples, dtype=int)

    for draw in range(config.samples):
        tau = generate_tau(hyperparameters.a1, hyperparameters.a2, config.p_max, rng)
        if config.model == "extended":
            lambda_values = generate_lambda_extended(
                hyperparameters.c1,
                hyperparameters.c2,
                hyperparameters.a,
                config.m,
                config.p_max,
                rng,
            )
        elif config.model == "original":
            lambda_values = generate_lambda_original(
                hyperparameters.a,
                config.m,
                config.p_max,
                rng,
            )
        else:
            raise ValueError("model must be either 'original' or 'extended'.")

        a_matrices = generate_a_matrices(lambda_values, tau, rng)
        p_matrices = a_stack_to_p_stack(a_matrices)
        orders[draw] = compute_optimal_p(
            p_matrices,
            m=config.m,
            n_observations=config.n_observations,
            beta=config.beta,
            p_max=config.p_max,
        )

    return orders

