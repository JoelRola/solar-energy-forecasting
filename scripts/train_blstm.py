"""Train the recovered bidirectional LSTM architecture."""

from _run_model import run
from solar_forecasting.models.blstm import build_model

if __name__ == "__main__":
    run("blstm", build_model)
