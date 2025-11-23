
"""Train Prophet model with external regressors and save a 90-day forecast.
Configurable via command-line arguments (changepoint_prior_scale, seasonality_mode, etc.).
"""
import argparse
import pandas as pd
from prophet import Prophet
import json
from pathlib import Path

def main(csv='data/timeseries_with_regressors.csv', cps=0.05, seasonality_mode='additive', seasonality_prior_scale=10.0, changepoint_range=0.8, periods=90):
    df = pd.read_csv(csv, parse_dates=['ds'])
    m = Prophet(changepoint_prior_scale=float(cps), seasonality_mode=seasonality_mode, seasonality_prior_scale=float(seasonality_prior_scale), changepoint_range=float(changepoint_range))
    # add regressors
    if 'temp' in df.columns:
        m.add_regressor('temp')
    if 'holiday' in df.columns:
        m.add_regressor('holiday')
    m.fit(df[['ds','y'] + [c for c in ['temp','holiday'] if c in df.columns]])
    future = m.make_future_dataframe(periods=int(periods))
    if 'temp' in df.columns:
        # simple extension of temp: repeat last seasonal pattern (user should replace with external regressor forecast)
        last_temp = df['temp'].values[-365:]
        reps = int(np.ceil(periods/len(last_temp)))
        future_temp = list(last_temp) * reps
        future_temp = future_temp[:len(future)]
        future['temp'] = future_temp
    if 'holiday' in df.columns:
        future['holiday'] = 0
    forecast = m.predict(future)
    Path('artifacts').mkdir(exist_ok=True)
    forecast.to_csv('artifacts/prophet_forecast.csv', index=False)
    with open('artifacts/prophet_config.json','w') as f:
        json.dump({'changepoint_prior_scale':cps, 'seasonality_mode':seasonality_mode, 'seasonality_prior_scale':seasonality_prior_scale, 'changepoint_range':changepoint_range}, f)
    print('Saved forecast to artifacts/prophet_forecast.csv and config to artifacts/prophet_config.json')

if __name__ == '__main__':
    import sys, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/timeseries_with_regressors.csv')
    parser.add_argument('--cps', default=0.05)
    parser.add_argument('--seasonality_mode', default='additive')
    parser.add_argument('--seasonality_prior_scale', default=10.0)
    parser.add_argument('--changepoint_range', default=0.8)
    parser.add_argument('--periods', default=90)
    args = parser.parse_args()
    main(csv=args.csv, cps=args.cps, seasonality_mode=args.seasonality_mode, seasonality_prior_scale=args.seasonality_prior_scale, changepoint_range=args.changepoint_range, periods=args.periods)
