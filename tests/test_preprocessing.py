import pandas as pd

from solar_forecasting.data import expand_daily_to_hourly, normalize_daily_schema
from solar_forecasting.preprocessing import (
    clean_hourly_frame,
    interpolate_hourly,
    prepare_energy_data,
)
from solar_forecasting.synthetic import make_synthetic_processed_data


def test_daily_schema_and_hour_expansion():
    raw = pd.DataFrame(
        {"Date & Time": ["01/01/2021"], "Daily Supply": [240], "Daily Demand ": [480]}
    )
    normalized = normalize_daily_schema(raw)
    hourly = expand_daily_to_hourly(raw)
    assert list(normalized.columns) == ["date", "daily_supply", "daily_demand"]
    assert len(hourly) == 24
    assert hourly["timestamp"].iloc[0].hour == 0
    assert hourly["timestamp"].iloc[-1].hour == 23


def test_clean_and_interpolate_hourly_frame():
    hourly = pd.DataFrame(
        {
            "datetime": pd.date_range("2021-01-01", periods=3, freq="h"),
            "date": pd.date_range("2021-01-01", periods=3, freq="D"),
            "daily_supply": [240.0, 480.0, 720.0],
            "daily_demand": [480.0, 960.0, 1440.0],
            "Weekly": [None, None, None],
            "Unnamed: 5": [None, None, None],
        }
    )
    clean = clean_hourly_frame(hourly)
    assert "Weekly" not in clean
    assert clean["hourly_supply"].tolist() == [10.0, 20.0, 30.0]
    clean.loc[1, "hourly_demand"] = float("nan")
    result = interpolate_hourly(clean)
    assert result["hourly_demand_interpolated"].iloc[1] == 40.0


def test_synthetic_prepare_pipeline(tmp_path):
    daily = pd.DataFrame(
        {
            "Date & Time": ["01/01/2024", "02/01/2024", "03/01/2024"],
            "Daily Supply": [240.0, 264.0, 288.0],
            "Daily Demand": [480.0, 504.0, 528.0],
        }
    )
    input_path = tmp_path / "synthetic_daily.csv"
    output_path = tmp_path / "processed" / "synthetic_processed.csv"
    daily.to_csv(input_path, index=False)
    result = prepare_energy_data(input_path, output_path)
    assert output_path.exists()
    assert len(result) == 72
    assert result["hourly_supply_interpolated"].notna().all()
    assert len(make_synthetic_processed_data(days=3)) == 72
