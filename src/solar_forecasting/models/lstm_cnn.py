"""Recovered CNN-LSTM hybrid architecture."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D


def build_model(lookback: int = 24, n_features: int = 2) -> Sequential:
    """Build Conv1D feature extraction followed by the recovered stacked LSTM."""
    return Sequential(
        [
            Input(shape=(lookback, n_features)),
            Conv1D(
                64,
                3,
                activation="relu",
                padding="same",
            ),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1),
        ],
        name="cnn_lstm_forecaster",
    )
