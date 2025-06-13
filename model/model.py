import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(12, activation='relu'),
        layers.Dense(6, activation='relu'),
        layers.Dense(3, activation='relu'),
        layers.Dense(6, activation='relu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
