from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_prior_orders(orders: np.ndarray, p_max: int, output_path: Path | None = None) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(orders, binwidth=1, discrete=True, color="#b8a8d9", edgecolor="black", ax=ax)
    ax.axvline(float(np.median(orders)), color="#17356f", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, p_max + 1)
    ax.set_xticks(range(0, p_max + 1))
    ax.set_title("Histogram of Optimal p")
    ax.set_xlabel("Optimal p")
    ax.set_ylabel("Frequency")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_posterior_mass(posterior_mass: np.ndarray, output_path: Path | None = None) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=np.arange(len(posterior_mass)), y=posterior_mass, color="#6fa8a9", ax=ax)
    ax.set_xlabel("Effective lag order")
    ax.set_ylabel("Posterior probability")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

