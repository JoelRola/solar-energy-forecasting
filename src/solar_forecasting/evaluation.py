"""Shared regression metrics and reproduced-results CSV output."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRIC_COLUMNS = (
    "model",
    "r2",
    "mae",
    "rmse",
    "mape",
    "training_rows",
    "test_rows",
    "lookback",
    "seed",
    "timestamp",
)


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, mape_epsilon: float = 1e-8
) -> dict[str, float]:
    """Return R², MAE, RMSE, and MAPE; MAPE omits near-zero actual values."""
    y_true = np.asarray(actual, dtype=float).reshape(-1)
    y_pred = np.asarray(predicted, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or not len(y_true):
        raise ValueError("actual and predicted must be non-empty arrays of equal length")
    valid = np.abs(y_true) > mape_epsilon
    mape = (
        float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)
        if valid.any()
        else float("nan")
    )
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape,
    }


def append_metric_record(
    path: str | Path,
    model: str,
    metrics: dict[str, float],
    training_rows: int,
    test_rows: int,
    lookback: int,
    seed: int,
) -> None:
    """Append one reproduced run to a standardized comparison CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model,
        **metrics,
        "training_rows": training_rows,
        "test_rows": test_rows,
        "lookback": lookback,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    row = pd.DataFrame([record], columns=METRIC_COLUMNS)
    row.to_csv(destination, mode="a", header=not destination.exists(), index=False)
