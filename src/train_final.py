"""Train final model given hyperparameter config saved as JSON (from tuner or baseline)."""
import json, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_builder import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def main(config_path='artifacts/best_kt.json', epochs=50):
    cfg = json.load(open(config_path))
    df = pd.read_csv('data/train.csv')
    X = df.drop(columns=['MedHouseVal'], errors='ignore')
    if 'MedHouseVal' not in df.columns and 'target' in df.columns:
        X = df.drop(columns=['target'])
    y = df['MedHouseVal'] if 'MedHouseVal' in df.columns else df['target']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    os.makedirs('artifacts', exist_ok=True)
    model = build_model(cfg)
    chk = ModelCheckpoint('artifacts/final_model.h5', save_best_only=True, monitor='loss')
    es = EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(Xs, y, epochs=epochs, batch_size=32, callbacks=[chk, es], validation_split=0.1)
    model.save('artifacts/final_model.h5')
    print('Saved final model to artifacts/final_model.h5')

if __name__=='__main__':
    main()
