from dataclasses import replace
from pathlib import Path
import re

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from src.config import MGPModel, MGPHyperparameters, PosteriorConfig, StanPriorConfig
from src.io import ensure_dir, save_posterior_mass
from src.plotting import plot_posterior_mass
from src.truncation import calculate_threshold, effective_order_from_draws
from src.var import generate_dataset, mean_center


STAN_DIR = Path(__file__).parent / "stan"
P_DRAW_PATTERN = re.compile(r"^P\[(\d+),(\d+),(\d+)\]$")


def stan_model_path(model: MGPModel) -> Path:
    if model == "original":
        return STAN_DIR / "multiplicative_gamma_original.stan"
    if model == "extended":
        return STAN_DIR / "multiplicative_gamma_extended.stan"
    raise ValueError("model must be 'original' or 'extended'.")


def stan_data(
    y: np.ndarray,
    config: PosteriorConfig,
    hyperparameters: MGPHyperparameters,
    stan_prior: StanPriorConfig,
) -> dict[str, object]:
    data: dict[str, object] = {
        "m": config.m,
        "p": config.p_max,
        "N": config.n_observations,
        "n_miss": 0,
        "ind_miss": [],
        "df": config.m + stan_prior.df_offset,
        "scale_diag": stan_prior.scale_diag,
        "scale_offdiag": stan_prior.scale_offdiag,
        "a1": hyperparameters.a1,
        "a2": hyperparameters.a2,
        "a": hyperparameters.a,
        "y": y,
    }
    if config.model == "extended":
        data["c1"] = hyperparameters.c1
        data["c2"] = hyperparameters.c2
    return data


def extract_p_draws(draws: pd.DataFrame) -> np.ndarray:
    p_columns = []
    for column in draws.columns:
        match = P_DRAW_PATTERN.match(column)
        if match:
            p_columns.append((tuple(int(value) for value in match.groups()), column))

    p_columns.sort(key=lambda item: item[0])
    if not p_columns:
        raise ValueError("No P[...] draw columns found in Stan output.")

    p_columns = [column for _, column in p_columns]
    return draws[p_columns].to_numpy()


def run_single_model(
    model_type: MGPModel,
    y: np.ndarray,
    config: PosteriorConfig,
    hyperparameters: MGPHyperparameters,
    stan_prior: StanPriorConfig,
    output_dir: Path,
) -> np.ndarray:
    model = CmdStanModel(stan_file=str(stan_model_path(model_type)))
    effective_config = replace(config, model=model_type)
    fit = model.sample(
        data=stan_data(y, effective_config, hyperparameters, stan_prior),
        chains=config.chains,
        parallel_chains=config.chains,
        iter_warmup=config.warmup,
        iter_sampling=config.sampling,
        adapt_delta=config.adapt_delta,
        max_treedepth=config.max_treedepth,
        inits=0 if config.zero_inits else None,
        show_progress=config.show_progress,
        seed=config.seed,
    )

    draws = fit.draws_pd()
    ensure_dir(output_dir / "draws")
    draws.to_csv(output_dir / "draws" / "draws.csv", index=False)

    threshold = calculate_threshold(config.m, config.n_observations, config.beta)
    posterior = effective_order_from_draws(
        extract_p_draws(draws),
        p_max=config.p_max,
        threshold=threshold,
    )
    ensure_dir(output_dir / "posteriors")
    save_posterior_mass(output_dir / "posteriors" / "posterior_mass.csv", posterior["pmf"])
    plot_posterior_mass(posterior["pmf"], output_dir / "posteriors" / "posterior_mass.png")
    return posterior["pmf"]


def run_posterior_experiment(
    config: PosteriorConfig,
    hyperparameters: MGPHyperparameters | None = None,
    stan_prior: StanPriorConfig | None = None,
) -> dict[MGPModel, list[np.ndarray]]:
    hyperparameters = hyperparameters or MGPHyperparameters()
    stan_prior = stan_prior or StanPriorConfig()
    rng = np.random.default_rng(config.seed)

    models = (
        ["original", "extended"]
        if config.model == "both"
        else [config.model]
    )
    results: dict[MGPModel, list[np.ndarray]] = {model_type: [] for model_type in models}

    for iteration in range(1, config.iterations + 1):
        sigma = np.eye(config.m)
        mu = np.zeros(config.m)
        dataset = generate_dataset(
            config.n_observations,
            config.m,
            config.true_order,
            sigma,
            mu,
            rng,
        )
        ensure_dir(config.data_dir)
        np.savez_compressed(
            config.data_dir
            / f"dataset_m_{config.m}_p_{config.true_order}_iteration_{iteration}.npz",
            y=np.asarray(dataset["y"]),
            mu=np.asarray(dataset["mu"]),
            sigma=np.asarray(dataset["sigma"]),
        )
        y = mean_center(np.asarray(dataset["y"]))

        for model_type in models:
            output_dir = config.results_dir / f"{model_type}_results" / f"iteration_{iteration}"
            results[model_type].append(
                run_single_model(
                    model_type,
                    y,
                    config,
                    hyperparameters,
                    stan_prior,
                    output_dir,
                )
            )

    return results
