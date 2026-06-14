from pathlib import Path
from typing import Annotated

import typer

from src.config import MGPModel, MGPHyperparameters, PosteriorConfig, PosteriorModel, PriorConfig, StanPriorConfig
from src.io import save_prior_orders
from src.plotting import plot_prior_orders
from src.posterior import run_posterior_experiment
from src.priors import simulate_prior_effective_orders

app = typer.Typer(no_args_is_help=True)
DEFAULT_HYPERPARAMETERS = MGPHyperparameters()
DEFAULT_PRIOR = PriorConfig()
DEFAULT_POSTERIOR = PosteriorConfig()
DEFAULT_STAN_PRIOR = StanPriorConfig()


@app.command()
def prior(
    model: Annotated[MGPModel, typer.Option(help="original or extended")] = DEFAULT_PRIOR.model,
    beta: Annotated[float, typer.Option(help="Truncation level.")] = DEFAULT_PRIOR.beta,
    m: Annotated[int, typer.Option(help="VAR dimension.")] = DEFAULT_PRIOR.m,
    n_observations: Annotated[int, typer.Option(help="Number of observations used in thresholding.")] = DEFAULT_PRIOR.n_observations,
    samples: Annotated[int, typer.Option(help="Number of draws from the prior.")] = DEFAULT_PRIOR.samples,
    p_max: Annotated[int, typer.Option(help="Maximum lag order under consideration.")] = DEFAULT_PRIOR.p_max,
    seed: Annotated[int, typer.Option(help="Random seed.")] = DEFAULT_PRIOR.seed,
    a1: Annotated[float, typer.Option(help="Global precision shape for lag 1.")] = DEFAULT_HYPERPARAMETERS.a1,
    a2: Annotated[float, typer.Option(help="Global precision shape for lags >= 2.")] = DEFAULT_HYPERPARAMETERS.a2,
    a: Annotated[float, typer.Option(help="Local precision shape/rate parameter.")] = DEFAULT_HYPERPARAMETERS.a,
    c1: Annotated[float, typer.Option(help="Extended MGP local rate shape.")] = DEFAULT_HYPERPARAMETERS.c1,
    c2: Annotated[float, typer.Option(help="Extended MGP local rate rate.")] = DEFAULT_HYPERPARAMETERS.c2,
    output_dir: Annotated[Path, typer.Option(help="Directory for prior CSV and plot outputs.")] = Path("results/prior"),
) -> None:
    config = PriorConfig(
        model=model,
        beta=beta,
        m=m,
        n_observations=n_observations,
        samples=samples,
        p_max=p_max,
        seed=seed,
    )
    hyperparameters = MGPHyperparameters(a1=a1, a2=a2, a=a, c1=c1, c2=c2)
    orders = simulate_prior_effective_orders(config, hyperparameters)
    csv_path = output_dir / f"{model}_orders.csv"
    plot_path = output_dir / f"{model}_orders.png"
    save_prior_orders(csv_path, orders)
    plot_prior_orders(orders, p_max, plot_path)
    typer.echo(f"Saved prior orders to {csv_path}")
    typer.echo(f"Saved plot to {plot_path}")


@app.command()
def posterior(
    model: Annotated[PosteriorModel, typer.Option(help="original, extended, or both")] = DEFAULT_POSTERIOR.model,
    m: Annotated[int, typer.Option(help="VAR dimension.")] = DEFAULT_POSTERIOR.m,
    true_order: Annotated[int, typer.Option(help="True lag order used to simulate data.")] = DEFAULT_POSTERIOR.true_order,
    n_observations: Annotated[int, typer.Option(help="Number of observations in each simulated dataset.")] = DEFAULT_POSTERIOR.n_observations,
    beta: Annotated[float, typer.Option(help="Truncation level.")] = DEFAULT_POSTERIOR.beta,
    p_max: Annotated[int, typer.Option(help="Maximum lag order fitted by the Stan model.")] = DEFAULT_POSTERIOR.p_max,
    iterations: Annotated[int, typer.Option(help="Number of independent simulated datasets.")] = DEFAULT_POSTERIOR.iterations,
    chains: Annotated[int, typer.Option(help="Number of Stan chains.")] = DEFAULT_POSTERIOR.chains,
    warmup: Annotated[int, typer.Option(help="Number of Stan warmup iterations.")] = DEFAULT_POSTERIOR.warmup,
    sampling: Annotated[int, typer.Option(help="Number of Stan sampling iterations.")] = DEFAULT_POSTERIOR.sampling,
    adapt_delta: Annotated[float, typer.Option(help="Target HMC acceptance probability.")] = DEFAULT_POSTERIOR.adapt_delta,
    max_treedepth: Annotated[int, typer.Option(help="Maximum HMC tree depth.")] = DEFAULT_POSTERIOR.max_treedepth,
    zero_inits: Annotated[bool, typer.Option(help="Initialize Stan parameters at zero on the unconstrained scale.")] = DEFAULT_POSTERIOR.zero_inits,
    show_progress: Annotated[bool, typer.Option(help="Show CmdStanPy chain progress bars.")] = DEFAULT_POSTERIOR.show_progress,
    seed: Annotated[int, typer.Option(help="Random seed.")] = DEFAULT_POSTERIOR.seed,
    data_dir: Annotated[Path, typer.Option(help="Directory for simulated datasets.")] = DEFAULT_POSTERIOR.data_dir,
    results_dir: Annotated[Path, typer.Option(help="Directory for posterior outputs.")] = DEFAULT_POSTERIOR.results_dir,
    a1: Annotated[float, typer.Option(help="Global precision shape for lag 1.")] = DEFAULT_HYPERPARAMETERS.a1,
    a2: Annotated[float, typer.Option(help="Global precision shape for lags >= 2.")] = DEFAULT_HYPERPARAMETERS.a2,
    a: Annotated[float, typer.Option(help="Local precision shape/rate parameter.")] = DEFAULT_HYPERPARAMETERS.a,
    c1: Annotated[float, typer.Option(help="Extended MGP local rate shape.")] = DEFAULT_HYPERPARAMETERS.c1,
    c2: Annotated[float, typer.Option(help="Extended MGP local rate rate.")] = DEFAULT_HYPERPARAMETERS.c2,
    df_offset: Annotated[int, typer.Option(help="Inverse-Wishart degrees-of-freedom offset.")] = DEFAULT_STAN_PRIOR.df_offset,
    scale_diag: Annotated[float, typer.Option(help="Inverse-Wishart scale matrix diagonal.")] = DEFAULT_STAN_PRIOR.scale_diag,
    scale_offdiag: Annotated[float, typer.Option(help="Inverse-Wishart scale matrix off-diagonal.")] = DEFAULT_STAN_PRIOR.scale_offdiag,
) -> None:
    config = PosteriorConfig(
        model=model,
        m=m,
        true_order=true_order,
        n_observations=n_observations,
        beta=beta,
        p_max=p_max,
        iterations=iterations,
        chains=chains,
        warmup=warmup,
        sampling=sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        zero_inits=zero_inits,
        show_progress=show_progress,
        seed=seed,
        data_dir=data_dir,
        results_dir=results_dir,
    )
    hyperparameters = MGPHyperparameters(a1=a1, a2=a2, a=a, c1=c1, c2=c2)
    stan_prior = StanPriorConfig(
        df_offset=df_offset,
        scale_diag=scale_diag,
        scale_offdiag=scale_offdiag,
    )
    results = run_posterior_experiment(config, hyperparameters, stan_prior)
    for model_type, pmfs in results.items():
        for iteration, pmf in enumerate(pmfs, start=1):
            typer.echo(f"{model_type}, iteration {iteration}: {pmf}")
