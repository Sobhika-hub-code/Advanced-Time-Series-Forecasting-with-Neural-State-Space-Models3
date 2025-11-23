
"""SARIMAX baseline using pmdarima's auto_arima to select orders and forecasting with exogenous regressors."""
import pandas as pd
import pmdarima as pm
import json
from pathlib import Path

def main(csv='data/timeseries_with_regressors.csv', periods=90):
    df = pd.read_csv(csv, parse_dates=['ds'])
    y = df['y'].values
    exog = df[['temp','holiday']].values if set(['temp','holiday']).issubset(df.columns) else None
    # use a train portion for speed
    train_n = int(len(df)*0.85)
    y_train = y[:train_n]
    exog_train = exog[:train_n] if exog is not None else None
    model = pm.auto_arima(y_train, exogenous=exog_train, seasonal=True, m=7, stepwise=True, suppress_warnings=True, max_p=3, max_q=3, max_P=2, max_Q=2)
    print('Selected order:', model.order, 'seasonal_order:', model.seasonal_order)
    Path('artifacts').mkdir(exist_ok=True)
    with open('artifacts/sarimax_selected.json','w') as f:
        json.dump({'order': str(model.order), 'seasonal_order': str(model.seasonal_order)}, f)
    # forecast
    exog_future = exog[train_n:train_n+periods] if exog is not None else None
    fc, confint = model.predict(n_periods=periods, exogenous=exog_future, return_conf_int=True)
    import numpy as np
    df_fc = pd.DataFrame({'yhat': fc})
    df_fc.to_csv('artifacts/sarimax_forecast.csv', index=False)
    print('Saved artifacts/sarimax_forecast.csv')

if __name__=='__main__':
    main()
