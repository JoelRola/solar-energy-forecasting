"""Train the recovered CNN-LSTM hybrid architecture."""

from _run_model import run
from solar_forecasting.models.lstm_cnn import build_model

if __name__ == "__main__":
    run("lstm_cnn", build_model)
