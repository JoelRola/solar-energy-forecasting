"""Train the recovered standalone Random Forest model."""

from _run_model import run
from solar_forecasting.models.random_forest import build_model

if __name__ == "__main__":
    run("random_forest", build_model, model_type="tree")
