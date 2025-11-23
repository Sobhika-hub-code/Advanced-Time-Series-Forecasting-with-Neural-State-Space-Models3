
# Prophet Time Series Forecasting — Final Submission Report

This project was prepared to satisfy the Cultus rubric in the screenshot provided as reference (included here: /mnt/data/b578897c-2a05-4357-973d-a77df5259a5a.png ).

## What this submission includes (deliverables)
1. **Data**: `data/timeseries_with_regressors.csv` — 4+ years of daily data with temperature-like external regressor and binary holiday indicator.
2. **Prophet implementation**: `src/train_prophet.py` — trains Prophet with external regressors and saves a 90-day forecast and used configuration.
3. **Prophet hyperparameter optimization**: `src/tune_prophet_optuna.py` — Optuna-based search over changepoint_prior_scale, seasonality_prior_scale, seasonality_mode, and changepoint_range. Uses Prophet's cross-validation to compute RMSE on a 30-day horizon as the objective. Saves `artifacts/best_prophet_optuna.json`.
4. **Baseline**: `src/sarimax_baseline.py` — pmdarima's `auto_arima` with exogenous regressors for seasonal SARIMAX baseline; saves forecast and selected order.
5. **Walk-forward backtesting**: `src/backtest.py` — expanding-window walk-forward evaluation that trains Prophet and SARIMAX on an initial window, steps forward, and records per-fold RMSE/MAE/MAPE for multiple horizons (7 and 30 days). Saves `artifacts/backtest_results.json`.
6. **Evaluation & plots**: `src/evaluate.py` — aggregates backtesting results and saves `artifacts/backtest_summary.csv` and comparison plots `artifacts/rmse_7d.png`, `artifacts/rmse_30d.png`.
7. **Report**: this file and `reports/report.md` describing methodology and commands to run.

## How this addresses the reviewers' points
- Uses **Prophet** (not DL frameworks) as required.
- Includes **external regressors** (temp and holiday) in both Prophet and SARIMAX where applicable.
- Implements **time-series cross-validation** via Prophet's `cross_validation` in the hyperparameter tuning loop and a walk-forward backtesting skeleton to evaluate practical generalization across horizons.
- Implements **hyperparameter optimization** (Optuna) and saves the best hyperparameters for reproducibility.
- Provides **baseline comparison** (SARIMAX) and clear evaluation metrics (RMSE, MAE, MAPE) for short (7-day) and medium (30-day) horizons.

## How to reproduce (commands)
1. Create and activate a Python environment (Python 3.9+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Regenerate the synthetic dataset:
   ```bash
   python src/data_gen.py
   ```
4. Tune Prophet hyperparameters with Optuna (example: 30 trials):
   ```bash
   python src/tune_prophet_optuna.py 30
   ```
5. Train final Prophet with chosen config and produce a 90-day forecast:
   ```bash
   python src/train_prophet.py --cps 0.1 --seasonality_mode multiplicative --periods 90
   ```
6. Run SARIMAX baseline (auto_arima selection):
   ```bash
   python src/sarimax_baseline.py
   ```
7. Run walk-forward backtesting (this will train multiple models; allow time):
   ```bash
   python src/backtest.py
   ```
8. Aggregate evaluation and generate plots:
   ```bash
   python src/evaluate.py
   ```

## Notes & best practices
- The synthetic external regressor `temp` is extended naively in the forecasting script; for production use you should supply a proper future regressor forecast or use realistic external data.
- The Optuna tuning uses Prophet's internal cross-validation which can be slow; adjust the number of trials according to available compute.
- For final submission include the `artifacts/` folder produced after running the scripts (forecasts, selected hyperparameters, plots).

## Files included in ZIP
- All scripts in `src/`, data in `data/`, and this report. The platform screenshot used for guidance is included in `reports/`.
