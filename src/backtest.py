"""Walk-forward evaluation skeleton: train small models on expanding windows and record RMSEs."""
import pandas as pd, numpy as np, json
from sklearn.model_selection import TimeSeriesSplit
from model_builder import build_model
from sklearn.preprocessing import StandardScaler
import os

def walk_forward(n_splits=5):
    df = pd.read_csv('data/train.csv')
    y = df['MedHouseVal'] if 'MedHouseVal' in df.columns else df['target']
    X = df.drop(columns=['MedHouseVal'], errors='ignore')
    if 'MedHouseVal' not in df.columns and 'target' in df.columns:
        X = df.drop(columns=['target'])
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[val_idx]
        scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xv_s = scaler.transform(Xv)
        hp = {'units':64,'n_layers':2,'dropout':0.1,'lr':1e-3}
        model = build_model(hp)
        model.fit(Xtr_s, ytr, epochs=10, batch_size=32, verbose=0)
        preds = model.predict(Xv_s).ravel()
        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(yv, preds, squared=False)
        results.append(rmse)
        print(f'Fold {fold} RMSE:', rmse)
    with open('artifacts/walk_forward_results.json','w') as f:
        json.dump({'rmses': results}, f)
    print('Saved artifacts/walk_forward_results.json')

if __name__=='__main__':
    walk_forward()
