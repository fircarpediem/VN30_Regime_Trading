# VN30 Regime Trading

A simple regime-based trading project for VN30 data using a Hidden Markov Model (HMM) for market state detection and `vectorbt` for backtesting.

## Overview

This project:
- Loads engineered VN30 features from `data/processed/vn30_features.csv`
- Trains a 3-state Gaussian HMM on:
  - `log_return`
  - `volatility`
- Estimates the probability of a selected bull regime
- Converts regime probabilities into discrete target portfolio weights
- Backtests the strategy with `vectorbt`

Current discrete weight logic in `HMMStrategy`:
- Bull regime probability >= 0.75 -> 100% allocation
- Bull regime probability >= 0.50 -> 50% allocation
- Otherwise -> 0% allocation

Signals are shifted by 1 bar to reduce lookahead bias.

## Project Structure

```text
vn30_regime_trading/
├── main.py
├── requirements.txt
├── backtest/
│   └── engine.py
├── data/
│   ├── processed/
│   │   └── vn30_features.csv
│   └── raw/
├── notebooks/
│   ├── data_exploration.ipynb
│   └── hmm_training.ipynb
└── src/
    └── strategy_logic.py
```

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt`:
  - pandas, numpy
  - vnstock
  - hmmlearn, scikit-learn
  - vectorbt
  - matplotlib, seaborn
  - nbformat>=4.2.0

## Setup

### Option 1: Conda

```bash
conda create -n vn30_quant python=3.10 -y
conda activate vn30_quant
pip install -r requirements.txt
```

### Option 2: venv

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Data Format

`data/processed/vn30_features.csv` must include at least:
- `time` (used as index)
- `close`
- `log_return`
- `volatility`

Example header:

```csv
time,open,high,low,close,volume,log_return,volatility
```

## Run Backtest

```bash
python main.py
```

