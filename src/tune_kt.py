"""Bayesian optimization with Keras-Tuner for Keras models."""
import os, json, argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import kerastuner as kt
from model_builder import build_model

def main(max_trials=20, executions_per_trial=1, seed=42):
    df = pd.read_csv('data/train.csv')
    X = df.drop(columns=['MedHouseVal'], errors='ignore')
    if 'MedHouseVal' not in df.columns and 'target' in df.columns:
        X = df.drop(columns=['target'])
    y = df['MedHouseVal'] if 'MedHouseVal' in df.columns else df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    os.makedirs('artifacts', exist_ok=True)
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='artifacts/kt_dir',
        project_name='kt_bayes'
    )
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_config = {k: best_hp.get(k) for k in ['units','n_layers','dropout','lr']}
    with open('artifacts/best_kt.json','w') as f:
        json.dump(best_config, f)
    print('Saved best hyperparameters to artifacts/best_kt.json')

if __name__ == '__main__':
    import sys
    max_trials = int(sys.argv[1]) if len(sys.argv)>1 else 20
    main(max_trials=max_trials)
