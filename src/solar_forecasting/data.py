"""CSV loading, historical schema normalization, and daily-to-hourly expansion."""

from pathlib import Path

import pandas as pd

DAILY_COLUMN_ALIASES = {
    "Date & Time": "date",
    "Daily Supply": "daily_supply",
    "Daily Demand": "daily_demand",
    "Daily Mismatch": "daily_mismatch",
}
REQUIRED_DAILY_COLUMNS = ("date", "daily_supply", "daily_demand")
MODEL_COLUMNS = ("hourly_supply_interpolated", "hourly_demand_interpolated")


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and normalize whitespace in its column names."""
    frame = pd.read_csv(Path(path))
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def normalize_daily_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the university CSV headings to stable snake_case names."""
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    normalized = normalized.rename(columns=DAILY_COLUMN_ALIASES)
    missing = [column for column in REQUIRED_DAILY_COLUMNS if column not in normalized]
    if missing:
        raise ValueError(f"Daily input is missing required columns: {missing}")
    normalized["date"] = pd.to_datetime(normalized["date"], dayfirst=True, errors="raise")
    normalized["date"] = normalized["date"].dt.normalize()
    return normalized


def expand_daily_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Repeat each daily record across its 24 hourly timestamps, as in the coursework."""
    daily = normalize_daily_schema(frame)
    if daily["date"].duplicated().any():
        raise ValueError("Daily input contains duplicate dates; expansion would duplicate hours")
    hourly_index = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max() + pd.Timedelta(hours=23),
        freq="h",
        name="timestamp",
    )
    hourly = pd.DataFrame({"timestamp": hourly_index})
    hourly["date"] = hourly["timestamp"].dt.normalize()
    daily_columns = [column for column in daily.columns if column != "date"]
    return hourly.merge(daily[["date", *daily_columns]], on="date", how="left").drop(
        columns="date"
    )


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """Load a processed time series and check its model feature columns."""
    frame = read_csv(path)
    if "timestamp" not in frame:
        raise ValueError("Processed data must contain a 'timestamp' column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    missing = [column for column in MODEL_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Processed data is missing model columns: {missing}")
    return frame.sort_values("timestamp").reset_index(drop=True)
