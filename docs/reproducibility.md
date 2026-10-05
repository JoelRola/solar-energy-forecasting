# Reproducibility

- Python: 3.11 recommended; project metadata allows Python 3.11 and 3.12.
- Install the declared runtime dependencies and the `dev` extra for pytest.
- Place an authorized `hourlydailyenergy.csv` at `data/raw/` and prepare processed data with the
  command in the main README.
- The default seed is 42 and is applied to Python, NumPy, and TensorFlow where supported.
- Scaler fitting on the full series and test-as-validation are intentionally retained from the
  coursework methodology and are documented evaluation limitations.
- TensorFlow may still exhibit device-dependent floating-point variation despite deterministic
  seed settings.
- The original package lock and full six-model result record were not recovered.
