"""Evaluation script computing RMSE, MAE, MAPE and plotting predictions."""
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

def main(model_path='artifacts/final_model.h5'):
    df_test = pd.read_csv('data/test.csv')
    X_test = df_test.drop(columns=['MedHouseVal'], errors='ignore')
    if 'MedHouseVal' not in df_test.columns and 'target' in df_test.columns:
        X_test = df_test.drop(columns=['target'])
    y_test = df_test['MedHouseVal'] if 'MedHouseVal' in df_test.columns else df_test['target']
    scaler_path = Path('artifacts/scaler_baseline.pkl')
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        Xs = scaler.transform(X_test)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(); Xs = scaler.fit_transform(X_test)
    model = load_model(model_path)
    preds = model.predict(Xs).ravel()
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    mape = (np.mean(np.abs((y_test - preds) / (np.clip(np.abs(y_test), 1e-8, None))))) * 100
    print('RMSE:', rmse, 'MAE:', mae, 'MAPE(%):', mape)
    # plot
    plt.figure(figsize=(8,4))
    plt.plot(y_test.values[:200], label='true')
    plt.plot(preds[:200], label='pred')
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/prediction_sample.png')
    print('Saved artifacts/prediction_sample.png')

if __name__=='__main__':
    main()
