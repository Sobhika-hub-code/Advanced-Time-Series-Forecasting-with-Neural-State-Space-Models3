"""Keras model builder for Keras-Tuner and training."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(hp):
    # hp can be a dict or keras_tuner HyperParameters object
    if hasattr(hp, 'Choice'):
        units = hp.Int('units', 32, 256, step=32)
        n_layers = hp.Int('n_layers', 1, 4)
        dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr = hp.Choice('lr', [1e-2, 1e-3, 1e-4])
    else:
        units = int(hp.get('units', 64))
        n_layers = int(hp.get('n_layers', 2))
        dropout = float(hp.get('dropout', 0.1))
        lr = float(hp.get('lr', 1e-3))

    inputs = keras.Input(shape=(8,), name='inputs')
    x = inputs
    for i in range(n_layers):
        x = layers.Dense(units, activation='relu')(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name='target')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model
