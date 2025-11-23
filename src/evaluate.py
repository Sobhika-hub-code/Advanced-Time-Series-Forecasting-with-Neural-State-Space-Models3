
"""Aggregate backtest results (artifacts/backtest_results.json) and produce a summary table and plots for 7-day and 30-day horizons.
"""
import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def summarize(results_path='artifacts/backtest_results.json'):
    r = json.load(open(results_path))
    summary = {'model':[], 'horizon':[], 'rmse_mean':[], 'mae_mean':[], 'mape_mean':[]}
    for model in r:
        for h in r[model]:
            df = pd.DataFrame(r[model][h])
            summary['model'].append(model)
            summary['horizon'].append(int(h))
            summary['rmse_mean'].append(df['rmse'].mean())
            summary['mae_mean'].append(df['mae'].mean())
            summary['mape_mean'].append(df['mape'].mean())
    sdf = pd.DataFrame(summary)
    Path('artifacts').mkdir(exist_ok=True)
    sdf.to_csv('artifacts/backtest_summary.csv', index=False)
    print('Saved artifacts/backtest_summary.csv')
    # plot comparison
    for h in sorted(sdf['horizon'].unique()):
        sub = sdf[sdf['horizon']==h]
        plt.figure(figsize=(6,3))
        plt.bar(sub['model'], sub['rmse_mean'])
        plt.title(f'RMSE comparison - {h}-day horizon')
        plt.tight_layout()
        plt.savefig(f'artifacts/rmse_{h}d.png')
    print('Saved comparison plots in artifacts/')

if __name__=='__main__':
    summarize()
