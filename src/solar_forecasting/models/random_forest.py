"""Recovered standalone Random Forest regressor."""

from sklearn.ensemble import RandomForestRegressor


def build_model(seed: int = 42) -> RandomForestRegressor:
    """Create the coursework Random Forest (100 trees, max depth 10)."""
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
