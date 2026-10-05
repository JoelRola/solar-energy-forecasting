"""Preparation of model-ready supply and demand time series."""

from pathlib import Path

import pandas as pd

from solar_forecasting.data import expand_daily_to_hourly, read_csv

INTERPOLATE_COLUMNS = ("hourly_supply", "hourly_demand", "hourly_mismatch")


def clean_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop date-only and spreadsheet artifact columns, then derive hourly allocations."""
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if "datetime" in cleaned and "timestamp" not in cleaned:
        cleaned = cleaned.rename(columns={"datetime": "timestamp"})
    if "timestamp" not in cleaned:
        raise ValueError("Hourly input must have a 'timestamp' or 'datetime' column")
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="raise")
    cleaned = cleaned.drop(columns="date", errors="ignore")
    artifacts = [c for c in cleaned if c.startswith("Unnamed") or "Weekly" in c]
    cleaned = cleaned.drop(columns=artifacts)
    for column in ("daily_supply", "daily_demand"):
        if column not in cleaned:
            raise ValueError(f"Hourly input is missing '{column}'")
    # Faithful to the recovered method: daily totals are allocated evenly over 24 hours.
    cleaned["hourly_supply"] = cleaned["daily_supply"] / 24.0
    cleaned["hourly_demand"] = cleaned["daily_demand"] / 24.0
    return cleaned.sort_values("timestamp").reset_index(drop=True)


def interpolate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate available hourly columns by row order, matching the legacy method."""
    result = frame.copy()
    columns = [name for name in INTERPOLATE_COLUMNS if name in result]
    if not columns:
        raise ValueError("No hourly_supply/hourly_demand/hourly_mismatch columns are available")
    for column in columns:
        result[f"{column}_interpolated"] = result[column].interpolate(method="linear")
    return result


def prepare_energy_data(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Read daily coursework data, expand it, allocate hourly values, interpolate, and save."""
    raw = read_csv(input_path)
    hourly = expand_daily_to_hourly(raw)
    cleaned = clean_hourly_frame(hourly)
    processed = interpolate_hourly(cleaned)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(destination, index=False)
    return processed
