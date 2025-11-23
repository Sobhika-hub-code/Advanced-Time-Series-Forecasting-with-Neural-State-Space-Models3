
Prophet Time Series Forecasting Submission
==========================================

This project implements the required deliverables for 'Advanced Time Series Forecasting with Prophet':
- Uses a multi-year daily time series with external regressors (temperature, holiday indicator)
- Implements Prophet model training and forecasting (90-day horizon by default)
- Implements Prophet hyperparameter optimization using Optuna (tune_prophet_optuna.py)
- Implements a SARIMAX baseline (sarimax_baseline.py) using pmdarima for automated order selection
- Implements walk-forward backtesting (backtest.py) comparing Prophet and SARIMAX across 7-day and 30-day horizons; computes RMSE/MAE/MAPE
- Provides evaluation scripts and a detailed report

Quick start (after creating a virtualenv and installing requirements):
  python src/data_gen.py
  python src/tune_prophet_optuna.py --trials 30
  python src/sarimax_baseline.py
  python src/backtest.py --horizons 7 30 --initial-window-days 365
  python src/evaluate.py

See reports/report.md for more details and explanations of hyperparameters and evaluation.
