import numpy as np
import pytest

from solar_forecasting.evaluation import regression_metrics


def test_regression_metrics():
    result = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert result["r2"] < 1
    assert result["mae"] == pytest.approx(1 / 3)
    assert result["rmse"] > result["mae"]
    assert result["mape"] > 0


def test_mape_omits_near_zero_actual_values():
    result = regression_metrics(np.array([0.0, 2.0]), np.array([100.0, 3.0]))
    assert result["mape"] == 50.0
