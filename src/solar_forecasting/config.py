"""Shared project defaults and repository-relative paths."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hourlydailyenergy.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "interpolated_energy.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "synthetic_energy.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "reproduced" / "metrics" / "model_comparison.csv"
FIGURES_PATH = PROJECT_ROOT / "results" / "reproduced" / "figures"


@dataclass(frozen=True)
class ForecastConfig:
    """Experiment settings retained from the recovered model scripts."""

    lookback: int = 24
    test_size: float = 0.2
    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    early_stopping_patience: int = 3


DEFAULT_CONFIG = ForecastConfig()
