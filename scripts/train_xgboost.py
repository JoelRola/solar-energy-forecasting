"""Train the recovered standalone XGBoost model."""

from _run_model import run
from solar_forecasting.models.xgboost_model import build_model

if __name__ == "__main__":
    run("xgboost", build_model, model_type="tree")
