"""24-hour windows, chronological splitting, and demand inverse scaling."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from solar_forecasting.data import MODEL_COLUMNS


@dataclass
class SequenceSplit:
    """Scaled windows and their chronological train/test labels."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    lookback: int

    def inverse_demand(self, values: np.ndarray) -> np.ndarray:
        """Convert scaled demand values back to the original demand units."""
        values = np.asarray(values).reshape(-1)
        dummy = np.zeros((len(values), len(MODEL_COLUMNS)), dtype=float)
        dummy[:, 1] = values
        return self.scaler.inverse_transform(dummy)[:, 1]


def create_sequences(values: np.ndarray, lookback: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Use the preceding lookback rows of both features to predict next-row demand."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("values must be a 2D array with supply and demand columns")
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if len(array) <= lookback:
        raise ValueError("at least lookback + 1 rows are required")
    X = np.stack([array[i : i + lookback] for i in range(len(array) - lookback)])
    y = array[lookback:, 1]
    return X, y


def chronological_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ordered windows without shuffling; the final fraction is the test set."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(X) != len(y) or len(X) < 2:
        raise ValueError("X and y must have the same length and at least two rows")
    split_at = int(np.floor(len(X) * (1.0 - test_size)))
    split_at = min(max(split_at, 1), len(X) - 1)
    return X[:split_at], X[split_at:], y[:split_at], y[split_at:]


def make_sequence_split(
    frame: pd.DataFrame, lookback: int = 24, test_size: float = 0.2
) -> SequenceSplit:
    """Scale the two features, generate windows, then split them chronologically.

    Scaling the complete series before splitting intentionally preserves the recovered
    coursework procedure. This leaks test-range information into scaling; see the methodology
    documentation before interpreting reproduced metrics as unbiased estimates.
    """
    missing = [column for column in MODEL_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Input is missing model feature columns: {missing}")
    values = frame.loc[:, MODEL_COLUMNS].astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = create_sequences(scaled, lookback=lookback)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size=test_size)
    return SequenceSplit(X_train, X_test, y_train, y_test, scaler, lookback)
