"""Recovered standalone XGBoost regressor; not an ensemble with Random Forest."""

from xgboost import XGBRegressor


def build_model(seed: int = 42) -> XGBRegressor:
    """Create the coursework XGBRegressor with its recovered settings."""
    return XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=seed)
