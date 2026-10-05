"""Deterministic toy data for tests and examples only."""

import numpy as np
import pandas as pd

from solar_forecasting.data import expand_daily_to_hourly
from solar_forecasting.preprocessing import clean_hourly_frame, interpolate_hourly


def make_synthetic_processed_data(days: int = 14) -> pd.DataFrame:
    """Create a small clearly synthetic series using the documented coursework pipeline."""
    if days < 2:
        raise ValueError("days must be at least 2")
    index = np.arange(days, dtype=float)
    supply = 240.0 + 18.0 * np.sin(index * 0.71) + index * 0.2
    demand = 480.0 + 26.0 * np.cos(index * 0.43) + index * 0.35
    daily = pd.DataFrame(
        {
            "Date & Time": pd.date_range("2024-01-01", periods=days, freq="D").strftime(
                "%d/%m/%Y"
            ),
            "Daily Supply": supply,
            "Daily Demand": demand,
            "Daily Mismatch": supply - demand,
        }
    )
    hourly = expand_daily_to_hourly(daily)
    return interpolate_hourly(clean_hourly_frame(hourly))
