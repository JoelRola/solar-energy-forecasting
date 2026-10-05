"""Recovered one-dimensional CNN architecture."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D


def build_model(lookback: int = 24, n_features: int = 2) -> Sequential:
    """Build the recovered two-convolution demand forecaster."""
    return Sequential(
        [
            Input(shape=(lookback, n_features)),
            Conv1D(64, 3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(32, 3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation="relu"),
            Dense(1),
        ],
        name="cnn_forecaster",
    )
