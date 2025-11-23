
"""Walk-forward backtesting comparing Prophet and SARIMAX.
Computes RMSE, MAE, MAPE for horizons (e.g., 7 and 30 days).
Saves detailed per-fold results to artifacts/backtest_results.json
"""
import pandas as pd, numpy as np, json, argparse
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
from pathlib import Path

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def evaluate_forecast(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    m = mape(y_true, y_pred)
    return {'rmse': float(rmse), 'mae': float(mae), 'mape': float(m)}

def main(horizons=[7,30], initial_window_days=365, step_days=90):
    df = pd.read_csv('data/timeseries_with_regressors.csv', parse_dates=['ds'])
    results = {'prophet':{}, 'sarimax':{}}
    n = len(df)
    start = initial_window_days
    fold = 0
    while start + max(horizons) <= n:
        train = df.iloc[:start]
        test = df.iloc[start:start+max(horizons)]
        # Prophet
        m = Prophet()
        if 'temp' in df.columns: m.add_regressor('temp')
        if 'holiday' in df.columns: m.add_regressor('holiday')
        m.fit(train[['ds','y'] + [c for c in ['temp','holiday'] if c in train.columns]])
        future = m.make_future_dataframe(periods=max(horizons), freq='D')
        # attach exogenous regressor values for future (copy last known pattern)
        if 'temp' in df.columns:
            future_temp = list(train['temp'].values[-365:])
            future_temp = (future_temp * int(np.ceil(len(future)/len(future_temp))))[:len(future)]
            future['temp'] = future_temp
        if 'holiday' in df.columns:
            future['holiday'] = 0
        fc = m.predict(future).set_index('ds')
        # SARIMAX baseline
        y_train = train['y'].values
        exog_train = train[['temp','holiday']].values if set(['temp','holiday']).issubset(train.columns) else None
        sar = pm.auto_arima(y_train, exogenous=exog_train, seasonal=True, m=7, stepwise=True, suppress_warnings=True, max_p=3, max_q=3, max_P=2, max_Q=2)
        exog_future = test[['temp','holiday']].values if set(['temp','holiday']).issubset(test.columns) else None
        sar_fc = sar.predict(n_periods=len(test), exogenous=exog_future)
        # Evaluate horizons
        for h in horizons:
            true = test['y'].values[:h]
            # prophet predictions: select first h days after train end
            prophet_pred = fc['yhat'].values[:h]
            sar_pred = sar_fc[:h]
            p_metrics = evaluate_forecast(true, prophet_pred)
            s_metrics = evaluate_forecast(true, sar_pred)
            results['prophet'].setdefault(str(h), []).append(p_metrics)
            results['sarimax'].setdefault(str(h), []).append(s_metrics)
        fold += 1
        start += step_days
    Path('artifacts').mkdir(exist_ok=True)
    with open('artifacts/backtest_results.json','w') as f:
        json.dump(results, f, indent=2)
    print('Saved artifacts/backtest_results.json')

if __name__=='__main__':
    import sys
    args = sys.argv[1:]
    # simple parsing
    main()
