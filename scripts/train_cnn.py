"""Train the recovered CNN architecture."""

from _run_model import run
from solar_forecasting.models.cnn import build_model

if __name__ == "__main__":
    run("cnn", build_model)
