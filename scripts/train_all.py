"""Run each recovered model independently, including separate RF and XGBoost fits."""

from solar_forecasting.config import PROCESSED_DATA_PATH
from solar_forecasting.models.blstm import build_model as build_blstm
from solar_forecasting.models.cnn import build_model as build_cnn
from solar_forecasting.models.lstm import build_model as build_lstm
from solar_forecasting.models.lstm_cnn import build_model as build_lstm_cnn
from solar_forecasting.models.random_forest import build_model as build_random_forest
from solar_forecasting.models.xgboost_model import build_model as build_xgboost
from solar_forecasting.runner import run_experiment


def main() -> None:
    experiments = [
        ("lstm", build_lstm, "neural"),
        ("blstm", build_blstm, "neural"),
        ("cnn", build_cnn, "neural"),
        ("lstm_cnn", build_lstm_cnn, "neural"),
        ("random_forest", build_random_forest, "tree"),
        ("xgboost", build_xgboost, "tree"),
    ]
    for name, builder, kind in experiments:
        print(f"\n=== {name} ===")
        run_experiment(name, builder, PROCESSED_DATA_PATH, model_type=kind)


if __name__ == "__main__":
    main()
