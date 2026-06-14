from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MGPModel = Literal["original", "extended"]
PosteriorModel = Literal["original", "extended", "both"]


@dataclass(frozen=True)
class MGPHyperparameters:
    a1: float = 2.5
    a2: float = 3.0
    a: float = 6.0
    c1: float = 10.0
    c2: float = 0.1


@dataclass(frozen=True)
class PriorConfig:
    model: MGPModel = "original"
    beta: float = 0.99
    m: int = 3
    n_observations: int = 10_000
    samples: int = 500
    p_max: int = 30
    seed: int = 123


@dataclass(frozen=True)
class PosteriorConfig:
    model: PosteriorModel = "both"
    m: int = 1
    # true_order: int = 12
    true_order: int = 7
    n_observations: int = 500
    beta: float = 0.99
    p_max: int = 20
    # iterations: int = 5
    iterations: int = 10_000
    chains: int = 10
    warmup: int = 5
    sampling: int = 20
    adapt_delta: float = 0.95
    max_treedepth: int = 12
    zero_inits: bool = True
    show_progress: bool = False
    seed: int = 123
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")


@dataclass(frozen=True)
class StanPriorConfig:
    df_offset: int = 4
    scale_diag: float = 1.0
    scale_offdiag: float = 0.0
