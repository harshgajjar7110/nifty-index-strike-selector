# Nifty Index Strike Selector - Instructions

## Overview
This repository contains a modular pipeline for selecting option strikes for the Nifty index. The project is broken into numbered modules for data processing, feature engineering, modeling, calibration, strike selection, backtesting, and live use.

## Prerequisites
- Python 3.10+ recommended
- Create a virtual environment before installing dependencies

## Install
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart - Run full pipeline
To run the entire pipeline end-to-end:

```bash
python run_pipeline.py
```

This script coordinates the modules in the repository in the default sequence.

## Module overview
- `module1_data_pipeline.py`: data ingestion and basic preprocessing
- `module2_features.py`: feature engineering and transformations
- `module3_garch.py`: volatility modeling (GARCH-related functions)
- `module4_model.py`: predictive model training and inference
- `module5_calibration.py`: calibration of model outputs to market data
- `module6_strikes.py`: strike selection logic and utilities
- `module7_backtest.py`: backtesting framework and performance metrics
- `module8_live.py`: glue for live deployment / live signals

## Running individual modules
- To run a single module for debugging, call it directly with Python, e.g.:

```bash
python module2_features.py
```

- Many modules expose functions; prefer importing them in an interactive session or small driver script for focused testing.

## Configuration and data
- The repository expects data files to be placed or referenced by the modules. Inspect each module's top-level constants or arguments for configurable paths.
- If you add config files, document them here and load them from environment variables or a simple `config.yaml`.

## Testing & linting
- There are no formal tests in the repo by default. Add a `tests/` folder and use `pytest` to run tests:

```bash
pip install pytest
pytest
```

## Development workflow
- Create feature branches for non-trivial changes
- Keep changes small and add unit tests where appropriate
- Run the full pipeline locally before opening a PR

## Contributing
- Open an issue describing the change you want to make
- Create a branch named `feature/short-description`
- Open a pull request and include a short description and tests

## License
Add a `LICENSE` file if you intend to open-source this project. If unspecified, ask the repo owner which license to use.

## Next steps
- Consider adding a `README.md` section that includes example outputs and sample data
- Add a `config.yaml` and document expected keys

If you'd like, I can add a `README` section, example config, or a small test harness next.
