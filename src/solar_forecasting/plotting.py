"""Figures for reproduced model evaluations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_forecast_figures(
    actual: np.ndarray, predicted: np.ndarray, model: str, output_dir: str | Path
) -> list[Path]:
    """Save actual/predicted, residual, and error-distribution figures."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    residual = actual - predicted
    safe = model.lower().replace(" ", "_").replace("/", "_")
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 4))
    shown = min(200, len(actual))
    ax.plot(actual[:shown], label="Actual", linewidth=1.5)
    ax.plot(predicted[:shown], label="Predicted", linestyle="--")
    ax.set(title=f"{model}: next-hour demand", xlabel="Test hour", ylabel="Demand (hourly units)")
    ax.legend()
    fig.tight_layout()
    path = folder / f"{safe}_actual_vs_predicted.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predicted, residual, alpha=0.55, s=16)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title=f"{model}: residuals", xlabel="Predicted demand", ylabel="Actual − predicted")
    fig.tight_layout()
    path = folder / f"{safe}_residuals.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=30, edgecolor="white")
    ax.set(title=f"{model}: prediction error", xlabel="Actual − predicted", ylabel="Count")
    fig.tight_layout()
    path = folder / f"{safe}_error_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)
    return outputs


def save_loss_figure(history: object, model: str, output_dir: str | Path) -> Path:
    """Save training and validation loss curves from a Keras History object."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    values = history.history
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values.get("loss", []), label="Training loss")
    ax.plot(values.get("val_loss", []), label="Validation loss")
    ax.set(title=f"{model}: training history", xlabel="Epoch", ylabel="MSE loss")
    ax.legend()
    fig.tight_layout()
    path = folder / f"{model.lower().replace('-', '_')}_training_loss.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_feature_importance(model: object, feature_names: list[str], output_path: str | Path) -> Path:
    """Save tree feature importances when the estimator exposes them."""
    importances = np.asarray(model.feature_importances_)
    if len(importances) != len(feature_names):
        raise ValueError("feature_names must match the estimator's feature count")
    order = np.argsort(importances)[-20:]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.asarray(feature_names)[order], importances[order])
    ax.set(title="Tree-model feature importance", xlabel="Importance", ylabel="24-hour input feature")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
