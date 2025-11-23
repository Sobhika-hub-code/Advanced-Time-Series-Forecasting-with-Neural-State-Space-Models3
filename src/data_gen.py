
"""Generate or load the synthetic daily time series with external regressors."""
import pandas as pd
from pathlib import Path
import argparse
import numpy as np

def generate(out='data/timeseries_with_regressors.csv', seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2019-01-01', periods=4*365+60, freq='D')
    n = len(dates)
    trend = 0.01 * np.arange(n)
    seasonal_year = 8 * np.sin(2 * np.pi * (np.arange(n) / 365.25))
    seasonal_week = 3 * np.sin(2 * np.pi * (pd.Series(dates).dt.dayofweek / 7.0))
    noise = np.random.normal(scale=1.5, size=n)
    temp = 20 + 10 * np.sin(2 * np.pi * (np.arange(n) / 365.25) + 0.2) + np.random.normal(scale=1.0, size=n)
    holiday = ((pd.Series(dates).dt.month == 12) & (pd.Series(dates).dt.day.isin([24,25,31]))).astype(int)
    y = 100 + trend + seasonal_year + seasonal_week + 0.6 * temp + 5 * holiday + noise
    df = pd.DataFrame({'ds':dates, 'y':y, 'temp':temp, 'holiday':holiday})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('Wrote', out, 'shape=', df.shape)

if __name__ == '__main__':
    generate()
