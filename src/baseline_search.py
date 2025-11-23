"""Baseline hyperparameter search using Randomized Grid (scikit-learn) over Keras models."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from model_builder import build_model
import joblib

def create_keras(hp):
    return build_model(hp)

def main(n_iter=20, cv=3, random_state=42):
    df = pd.read_csv('data/train.csv')
    X = df.drop(columns=['MedHouseVal'], errors='ignore')
    if 'MedHouseVal' not in df.columns and 'target' in df.columns:
        X = df.drop(columns=['target'])
    y = df['MedHouseVal'] if 'MedHouseVal' in df.columns else df['target']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, 'artifacts/scaler_baseline.pkl')
    def builder(units=64, n_layers=2, dropout=0.1, lr=1e-3):
        hp = {'units': units, 'n_layers': n_layers, 'dropout': dropout, 'lr': lr}
        return build_model(hp)
    reg = KerasRegressor(build_fn=builder, verbose=0)
    param_dist = {
        'units': [32, 64, 128],
        'n_layers': [1,2,3],
        'dropout': [0.0, 0.1, 0.2],
        'lr': [1e-2, 1e-3, 1e-4],
        'epochs': [20],
        'batch_size': [32, 64]
    }
    cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(reg, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, verbose=2)
    search.fit(Xs, y)
    print('Best params', search.best_params_)
    with open('artifacts/baseline_best.json','w') as f:
        json.dump(search.best_params_, f)

if __name__ == '__main__':
    main()
