"""Recovered stacked LSTM architecture."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM


def build_model(lookback: int = 24, n_features: int = 2) -> Sequential:
    """Build LSTM(64) → dropout → LSTM(32) → demand output."""
    return Sequential(
        [
            Input(shape=(lookback, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1),
        ],
        name="lstm_forecaster",
    )
