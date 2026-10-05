import numpy as np
import pandas as pd
import pytest

from solar_forecasting.sequences import (
    chronological_split,
    create_sequences,
    make_sequence_split,
)


def test_24_hour_sequence_targets_next_row():
    values = np.column_stack([np.arange(30), np.arange(100, 130)])
    X, y = create_sequences(values, lookback=24)
    assert X.shape == (6, 24, 2)
    assert y[0] == 124
    assert X[0, -1, 1] == 123


def test_split_is_chronological():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    x_train, x_test, y_train, y_test = chronological_split(X, y, 0.2)
    assert y_train.tolist() == list(range(8))
    assert y_test.tolist() == [8, 9]
    assert x_train[-1, 0] < x_test[0, 0]


def test_make_split_has_two_features_and_inverse_scaling():
    frame = pd.DataFrame(
        {
            "hourly_supply_interpolated": np.arange(60, dtype=float),
            "hourly_demand_interpolated": np.arange(100, 160, dtype=float),
        }
    )
    split = make_sequence_split(frame, lookback=24, test_size=0.2)
    assert split.X_train.shape[1:] == (24, 2)
    assert split.X_test.shape[1:] == (24, 2)
    assert split.inverse_demand(split.y_test).shape == split.y_test.shape


def test_too_short_sequence_is_rejected():
    with pytest.raises(ValueError):
        create_sequences(np.ones((24, 2)), lookback=24)
