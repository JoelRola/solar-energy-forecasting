"""Recovered bidirectional LSTM architecture."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM


def build_model(lookback: int = 24, n_features: int = 2) -> Sequential:
    """Build bidirectional LSTM(64) → dropout → bidirectional LSTM(32) → output."""
    return Sequential(
        [
            Input(shape=(lookback, n_features)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dense(1),
        ],
        name="blstm_forecaster",
    )
