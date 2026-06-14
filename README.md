Python implementation of Masters Thesis - A Bayesian Approach to Lag Selection in Higher-Order Stationary Vector Autoregressions 

## Setup

```bash
uv sync
```

CmdStan is required for posterior sampling. After setup:

```bash
uv run python -m cmdstanpy.install_cmdstan
```

## Run

Prior simulation:

```bash
uv run thesis prior --model original
uv run thesis prior --model extended
```

Prior simulation with all default parameters shown:

```bash
uv run thesis prior \
  --model original \
  --beta 0.99 \
  --m 3 \
  --n-observations 10000 \
  --samples 500 \
  --p-max 30 \
  --seed 123 \
  --a1 2.5 \
  --a2 3.0 \
  --a 6.0 \
  --c1 10.0 \
  --c2 0.1 \
  --output-dir results/prior
```

```bash
uv run thesis prior \
  --model extended \
  --beta 0.99 \
  --m 3 \
  --n-observations 10000 \
  --samples 500 \
  --p-max 30 \
  --seed 123 \
  --a1 2.5 \
  --a2 3.0 \
  --a 6.0 \
  --c1 10.0 \
  --c2 0.1 \
  --output-dir results/prior
```

Posterior simulation:

```bash
uv run thesis posterior --model both
```

Posterior simulation with all default parameters shown:

```bash
uv run thesis posterior \
  --model both \
  --m 1 \
  --true-order 7 \
  --n-observations 500 \
  --beta 0.99 \
  --p-max 20 \
  --iterations 10000 \
  --chains 1 \
  --warmup 1 \
  --sampling 20 \
  --adapt-delta 0.95 \
  --max-treedepth 12 \
  --zero-inits \
  --no-show-progress \
  --seed 123 \
  --data-dir data \
  --results-dir results \
  --a1 2.5 \
  --a2 3.0 \
  --a 6.0 \
  --c1 10.0 \
  --c2 0.1 \
  --df-offset 4 \
  --scale-diag 1.0 \
  --scale-offdiag 0.0
```

`--iterations` is the number of independent simulated datasets. It is not the
number of HMC samples; `--sampling` controls Stan sampling iterations.

## Structure

- `src/priors.py`: multiplicative gamma process prior simulation.
- `src/mappings.py`: transformations between unconstrained matrices, partial autocorrelations, and VAR coefficients.
- `src/var.py`: stationary VAR simulation utilities.
- `src/truncation.py`: effective lag-order thresholding.
- `src/posterior.py`: CmdStanPy posterior workflow.
- `src/plotting.py`: Seaborn plots.
- `src/stan/`: original Stan model files.

Posterior outputs are written under:

```text
results/
  original_results/
    iteration_1/
      draws/draws.csv
      posteriors/posterior_mass.csv
      posteriors/posterior_mass.png
  extended_results/
    iteration_1/
      draws/draws.csv
      posteriors/posterior_mass.csv
      posteriors/posterior_mass.png
```


