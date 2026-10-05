"""CLI helper shared by individual model entry points."""

import argparse
from collections.abc import Callable

from solar_forecasting.config import PROCESSED_DATA_PATH
from solar_forecasting.runner import run_experiment


def run(model_name: str, builder: Callable[[], object], model_type: str = "neural") -> None:
    parser = argparse.ArgumentParser(description=f"Train and evaluate {model_name}.")
    parser.add_argument("--data", default=str(PROCESSED_DATA_PATH), help="Processed input CSV")
    args = parser.parse_args()
    metrics = run_experiment(model_name, builder, args.data, model_type=model_type)
    print(f"{model_name} test metrics: {metrics}")
