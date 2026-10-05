"""Common experiment runner for recovered neural and tree models."""

import random
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from solar_forecasting.config import DEFAULT_CONFIG, FIGURES_PATH, METRICS_PATH, ForecastConfig
from solar_forecasting.data import load_processed_data
from solar_forecasting.evaluation import append_metric_record, regression_metrics
from solar_forecasting.plotting import (
    save_feature_importance,
    save_forecast_figures,
    save_loss_figure,
)
from solar_forecasting.sequences import make_sequence_split


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds where supported."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def run_experiment(
    model_name: str,
    builder: Callable[[], object],
    data_path: str | Path,
    config: ForecastConfig = DEFAULT_CONFIG,
    model_type: str = "neural",
    metrics_path: str | Path = METRICS_PATH,
    figures_path: str | Path = FIGURES_PATH,
) -> dict[str, float]:
    """Fit one model and save standardized test metrics and figures."""
    set_random_seeds(config.seed)
    frame = load_processed_data(data_path)
    split = make_sequence_split(frame, config.lookback, config.test_size)
    model = builder()

    if model_type == "neural":
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(
            split.X_train,
            split.y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(split.X_test, split.y_test),
            callbacks=[EarlyStopping(patience=config.early_stopping_patience)],
            verbose=1,
        )
        save_loss_figure(history, model_name, figures_path)
        train_X, test_X = split.X_train, split.X_test
    elif model_type == "tree":
        train_X = split.X_train.reshape(len(split.X_train), -1)
        test_X = split.X_test.reshape(len(split.X_test), -1)
        model.fit(train_X, split.y_train)
        names = [
            f"{feature}_t-{step}"
            for step in range(config.lookback, 0, -1)
            for feature in ("hourly_supply", "hourly_demand")
        ]
        if hasattr(model, "feature_importances_"):
            save_feature_importance(
                model, names, Path(figures_path) / f"{model_name}_feature_importance.png"
            )
    else:
        raise ValueError("model_type must be 'neural' or 'tree'")

    predictions = np.asarray(model.predict(test_X)).reshape(-1)
    actual = split.inverse_demand(split.y_test)
    predicted = split.inverse_demand(predictions)
    metrics = regression_metrics(actual, predicted)
    append_metric_record(
        metrics_path,
        model_name,
        metrics,
        len(split.X_train),
        len(split.X_test),
        config.lookback,
        config.seed,
    )
    save_forecast_figures(actual, predicted, model_name, figures_path)
    return metrics
