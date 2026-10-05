"""Train the recovered LSTM architecture."""

from _run_model import run
from solar_forecasting.models.lstm import build_model

if __name__ == "__main__":
    run("lstm", build_model)
