
"""Improved Optuna-based hyperparameter search for Prophet.
Saves best params to artifacts/best_prophet_optuna.json
"""
import optuna, json, argparse, numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

def objective(trial):
    cps = trial.suggest_float('changepoint_prior_scale', 1e-4, 1.0, log=True)
    seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 1e-2, 20.0, log=True)
    seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive','multiplicative'])
    changepoint_range = trial.suggest_float('changepoint_range', 0.5, 0.95)

    df = pd.read_csv('data/timeseries_with_regressors.csv', parse_dates=['ds'])
    m = Prophet(
        changepoint_prior_scale=cps,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        changepoint_range=changepoint_range
    )

    for reg in ['temp','holiday']:
        if reg in df.columns:
            m.add_regressor(reg)

    m.fit(df[['ds','y'] + [c for c in ['temp','holiday'] if c in df.columns]])

    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='30 days')
    perf = performance_metrics(df_cv)
    rmse = float(perf['rmse'].mean())
    return rmse

def main(trials=30):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=int(trials))
    best = study.best_params
    print('Best params:', best)

    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/best_prophet_optuna.json','w') as f:
        json.dump(best, f)

if __name__ == '__main__':
    import sys
    t = int(sys.argv[1]) if len(sys.argv)>1 else 30
    main(trials=t)
