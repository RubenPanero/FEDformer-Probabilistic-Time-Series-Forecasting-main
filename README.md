<div align="center">

<img src="docs/assets/readme_wordmark.svg" alt="FEDformer Probabilistic Forecasting" width="720">

# FEDformer Probabilistic Time-Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)]()
[![CI](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ci.yml)
[![Ruff](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ruff.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/ruff.yml)
[![Pylint](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/pylint.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/pylint.yml)
[![Security](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/security.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/security.yml)
[![Compatibility](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/compatibility.yml/badge.svg)](https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting/actions/workflows/compatibility.yml)

Probabilistic forecasting with `FEDformer`, `Normalizing Flows`, walk-forward
evaluation, conformal calibration, specialist export, and Optuna search.

[Quick Start](#quick-start) • [Inference](#inference) • [Results](#results-and-limitations) • [Testing](#testing-and-reproducibility)

</div>

![Repository Hero](docs/assets/fan_chart_nvda.png)

## Table of Contents

- [Overview](#overview)
- [Who Is This For?](#who-is-this-for)
- [Visual Demo](#visual-demo)
- [Quick Start](#quick-start)
- [Outputs](#outputs)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Inference](#inference)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Results and Limitations](#results-and-limitations)
- [Testing and Reproducibility](#testing-and-reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

## Overview

This repository packages an uncertainty-aware forecasting pipeline built around
`FEDformer` plus `Normalizing Flows`. It is designed for use cases where a
point forecast is not enough and downstream consumers need calibrated bands,
sampled scenarios, and risk-aware evaluation.

What it provides:

- Walk-forward backtesting with anti-leakage temporal splits
- Probabilistic outputs with explicit quantiles and sampling-based summaries
- Financial evaluation with Sharpe, Sortino, drawdown, volatility, VaR, and CVaR
- Canonical specialist export and reusable inference CLI
- Optuna-based hyperparameter search with persistent studies

| Snapshot | Value |
|----------|-------|
| Canonical specialists | `NVDA`, `GOOGL` |
| Python | `3.10`, `3.11` |
| Runtime | Single-process CPU/GPU training |
| Parallelism today | Multi-seed runs and Optuna subprocess trials |

## Who Is This For?

This repository is a good fit if you need:

- probabilistic forecasts instead of point-only predictions
- walk-forward evaluation that respects temporal ordering
- finance-oriented metrics such as Sharpe, Sortino, drawdown, VaR, and CVaR
- reusable specialist checkpoints for repeated inference workflows

Typical users:

- quantitative researchers testing distribution-aware signals
- ML practitioners working on uncertainty-aware forecasting pipelines
- engineers building repeatable backtesting and inference workflows around tabular time series

## Visual Demo

| Fan Chart | Calibration |
|-----------|-------------|
| Probabilistic band (`p10-p90`), median path (`p50`), and ground truth. | Reliability view plus PIT histogram for probabilistic quality checks. |
| ![Fan Chart NVDA](docs/assets/fan_chart_nvda.png) | ![Calibration NVDA](docs/assets/calibration_nvda.png) |

> No live demo is hosted right now. The intended evaluation path is local:
> train, export a canonical specialist, and inspect the generated CSVs and
> plots.

## Quick Start

### 1. Install

```bash
git clone https://github.com/RubenPanero/FEDformer-Probabilistic-Time-Series-Forecasting.git
cd FEDformer-Probabilistic-Time-Series-Forecasting

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run a canonical training job

```bash
MPLBACKEND=Agg python3 main.py \
  --csv data/NVDA_features.csv \
  --targets "Close" \
  --seq-len 96 \
  --pred-len 20 \
  --batch-size 64 \
  --splits 4 \
  --return-transform log_return \
  --metric-space returns \
  --gradient-clip-norm 0.5 \
  --seed 7 \
  --save-results \
  --save-canonical \
  --no-show
```

### 3. Run inference on a saved specialist

```bash
python3 -m inference \
  --ticker NVDA \
  --csv data/NVDA_features.csv \
  --plot \
  --output-dir results/
```

## Outputs

Typical artifacts from a training run:

| Artifact | Purpose |
|----------|---------|
| `predictions_*.csv` | Forecasts exported from training |
| `risk_metrics_*.csv` | VaR, CVaR, and related risk outputs |
| `portfolio_metrics_*.csv` | Strategy-level performance metrics |
| `training_history_*.csv` | Fold and epoch-level training history |
| `run_manifest_*.json` | Run metadata and aggregate metrics |
| `probabilistic_metrics_*.csv` | Coverage, pinball, calibration-style metrics |
| plots in `results/` | Fan chart and calibration views during inference |

If `--save-canonical` succeeds, the model can be reused later through the
inference CLI together with its saved preprocessing artifacts.

## Architecture

![Architecture Diagram](docs/assets/architecture.svg)

High-level flow:

```text
CSV -> TimeSeriesDataset / PreprocessingPipeline
    -> WalkForwardTrainer
        -> Flow_FEDformer
            -> FEDformer encoder/decoder
            -> Normalizing Flows
    -> ForecastOutput
    -> RiskSimulator + PortfolioSimulator
```

Core modules:

- `main.py`: training, evaluation, result writing, canonical export
- `config.py`: `FEDformerConfig` and grouped settings
- `models/`: FEDformer backbone and flow layers
- `training/`: trainer, scheduling, checkpoints, `ForecastOutput`
- `data/`: datasets, preprocessing, financial dataset helpers
- `inference/`: specialist loading and prediction CLI
- `simulations/`: risk and portfolio simulation
- `utils/`: metrics, plotting, experiment I/O
- `tune_hyperparams.py`: Optuna search runner
- `scripts/`: automation and verification helpers
- `tests/`: pytest suite

## Installation

### System assumptions

This project is documented and validated primarily on Linux.

Recommended prerequisites:

- Python `3.10` or `3.11`
- `venv`
- a working C or C++ build toolchain if a dependency needs compilation
- optional CUDA-capable GPU for faster training

### Environment variables

For Alpha Vantage-backed dataset downloads, export:

```bash
export ALPHA_VANTAGE_API_KEY="your_api_key"
```

The runtime reads `ALPHA_VANTAGE_API_KEY` directly from the environment. `.env`
files are not auto-loaded by the code.

## Usage

### Canonical training runs

NVDA:

```bash
MPLBACKEND=Agg python3 main.py \
  --csv data/NVDA_features.csv \
  --targets "Close" \
  --seq-len 96 \
  --pred-len 20 \
  --batch-size 64 \
  --splits 4 \
  --return-transform log_return \
  --metric-space returns \
  --gradient-clip-norm 0.5 \
  --seed 7 \
  --save-results \
  --save-canonical \
  --no-show
```

GOOGL:

```bash
MPLBACKEND=Agg python3 main.py \
  --csv data/GOOGL_features.csv \
  --targets "Close" \
  --seq-len 96 \
  --pred-len 20 \
  --batch-size 64 \
  --splits 4 \
  --return-transform log_return \
  --metric-space returns \
  --gradient-clip-norm 0.5 \
  --seed 7 \
  --save-results \
  --save-canonical \
  --no-show
```

### Advanced training example

```bash
MPLBACKEND=Agg python3 main.py \
  --csv data/financial_data.csv \
  --targets "close_price" \
  --date-col "date" \
  --seq-len 96 \
  --label-len 48 \
  --pred-len 24 \
  --e-layers 3 \
  --d-layers 1 \
  --n-flow-layers 4 \
  --flow-hidden-dim 64 \
  --dropout 0.1 \
  --epochs 15 \
  --batch-size 64 \
  --splits 5 \
  --use-checkpointing \
  --grad-accum-steps 1 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --scheduler-type cosine \
  --seed 123 \
  --deterministic \
  --save-results \
  --no-show
```

### Hyperparameter search

```bash
python3 tune_hyperparams.py \
  --csv data/NVDA_features.csv \
  --n-trials 8 \
  --n-splits 4 \
  --storage-path optuna_studies/nvda_smoke.db \
  --study-objective sharpe
```

The current Optuna search space includes:

- `seq_len`
- `pred_len`
- `batch_size`
- `gradient_clip_norm`
- `e_layers`
- `d_layers`
- `n_flow_layers`
- `flow_hidden_dim`
- `label_len`
- `dropout`

<details>
<summary>Useful helper scripts</summary>

- `python3 scripts/run_multi_seed.py --csv data/NVDA_features.csv --targets Close --seeds 7 42 123 --extra-args --seq-len 96 --pred-len 20 --batch-size 64`
- `python3 scripts/run_ablation_matrix.py`
- `python3 scripts/verify_cp_walkforward.py --report-only`
- `python3 scripts/validate_forecast.py --help`

</details>

## Inference

The public inference surface is function-based and CLI-accessible.

Examples:

```bash
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --output results/preds.csv
python3 -m inference --ticker NVDA --csv data/NVDA_features.csv --plot --output-dir results/
python3 -m inference --list-models
```

Inference notes:

- Canonical inference requires a checkpoint produced with `--save-canonical`
- preprocessing must be restored, not re-fit, during inference
- default generated plot names use the lowercase ticker inside the selected
  `--output-dir`
- the repository currently ships canonical specialists for `NVDA` and `GOOGL`

## Data Format

Expected CSV characteristics:

- one time column
- one or more numeric feature columns
- target columns selected with `--targets`

Typical financial datasets include fields such as:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Example:

```csv
date,close_price,volume,volatility,rsi
2023-01-01,100.5,1000000,0.15,45.2
2023-01-02,101.2,1200000,0.18,47.8
```

To build a market dataset:

```bash
python3 -m data.financial_dataset_builder --symbol NVDA --output_dir data --use_mock
```

Without `--use_mock`, the dataset builder uses Alpha Vantage and expects
`ALPHA_VANTAGE_API_KEY` to be set first.

## Configuration

There are two relevant configuration surfaces:

- CLI defaults in `main.py`
- constructor defaults in `FEDformerConfig`

They are not identical by design. CLI runs use larger, more practical defaults
than the raw config object.

Useful CLI groups:

- Data and splits: `--csv`, `--targets`, `--date-col`, `--seq-len`,
  `--label-len`, `--pred-len`, `--splits`, `--batch-size`
- Optimization: `--learning-rate`, `--weight-decay`, `--scheduler-type`,
  `--warmup-epochs`, `--patience`, `--min-delta`, `--gradient-clip-norm`
- Runtime: `--use-checkpointing`, `--grad-accum-steps`, `--deterministic`,
  `--compile-mode`, `--mc-dropout-eval-samples`
- Fine-tuning: `--finetune-from`, `--freeze-backbone`, `--finetune-lr`
- Output: `--save-fig`, `--save-results`, `--save-canonical`, `--no-show`
- Probabilistic evaluation: `--return-transform`, `--metric-space`,
  `--conformal-calibration`, `--cp-walkforward`

Defaults worth calling out:

- CLI defaults for sequence lengths: `seq_len=96`, `label_len=48`,
  `pred_len=24`
- Direct `FEDformerConfig` defaults: `seq_len=10`, `label_len=5`, `pred_len=5`
- CLI default for `--grad-accum-steps`: `1`
- Direct `FEDformerConfig` default for `gradient_accumulation_steps`: `2`
- Direct `FEDformerConfig` default for `mc_dropout_eval_samples`: `20`
- Odd `pred_len` values are accepted, but even values are strongly recommended
  for affine coupling stability
- Preset `fourier_optimized` keeps defaults intact but provides `modes=48` for
  opt-in performance experiments

Authoritative help:

```bash
python3 main.py --help
python3 tune_hyperparams.py --help
python3 -m inference --help
```

## Results and Limitations

Canonical benchmark results currently documented for `seed=7`:

| Ticker | Sharpe | Sortino | Max Drawdown |
|--------|--------|---------|--------------|
| NVDA   | +0.990 | +1.857  | -54.2%       |
| GOOGL  | +0.737 | +1.009  | -40.2%       |

Shared canonical setup:

- `seq_len=96`
- `pred_len=20`
- `batch_size=64`
- `splits=4`
- `return_transform=log_return`
- `metric_space=returns`
- `gradient_clip_norm=0.5`
- `seed=7`

Current probabilistic output behavior:

- explicit quantiles are carried through the pipeline
- the current default quantile set is `p10 / p50 / p90`
- `preds_*` remains the p50 compatibility path
- inference may additionally export sample mean summaries when MC samples are
  available

Current limitations:

- no distributed training implementation
- `pred_len` is best kept even for affine coupling compatibility
- walk-forward conformal calibration is only meaningful when enough folds exist
- `--cp-walkforward` with very small `--splits` can legitimately produce `0`
  calibrated folds

## Testing and Reproducibility

Fast local validation:

```bash
pytest -q -m "not slow"
```

Full test suite:

```bash
pytest -q
```

Lint and CI-parity checks:

```bash
ruff check .
ruff format --check .
make ci-check
```

Recommended smoke checks:

```bash
python3 main.py --help
python3 tune_hyperparams.py --help
python3 -m inference --help
python3 scripts/verify_cp_walkforward.py --help
```

Reproducibility notes:

- use `--seed <N>` and optionally `--deterministic`
- the canonical repository benchmark seed is `7`
- `MPLBACKEND=Agg` is recommended for headless Linux runs

## Contributing

Contributions are welcome.

Before opening a PR:

1. Run `make ci-check`
2. Keep changes focused and documented
3. Update `README.md` when changing public CLI behavior
4. Add or update tests for runtime-facing changes

Suggested PR contents:

- motivation
- summary of changes
- testing evidence
- notes on backward compatibility when relevant

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Credits

Built on top of the FEDformer forecasting approach and extended here with
probabilistic modeling, walk-forward evaluation, calibration workflows, and
specialist inference utilities.

References:

- Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for
  Long-term Series Forecasting"
- Dinh et al., "Density Estimation Using Real NVP"

Maintainer: Ruben Panero
