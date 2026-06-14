from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_prior_orders(path: Path, orders: np.ndarray) -> None:
    ensure_dir(path.parent)
    pd.DataFrame({"optimal_p": orders}).to_csv(path, index=False)


def save_posterior_mass(path: Path, posterior_mass: np.ndarray) -> None:
    ensure_dir(path.parent)
    pd.DataFrame({"p": np.arange(len(posterior_mass)), "probability": posterior_mass}).to_csv(
        path,
        index=False,
    )

