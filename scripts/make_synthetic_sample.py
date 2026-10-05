"""Generate the tracked, non-private synthetic example CSV."""

from solar_forecasting.config import SAMPLE_DATA_PATH
from solar_forecasting.synthetic import make_synthetic_processed_data


def main() -> None:
    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = make_synthetic_processed_data()
    frame.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {len(frame)} synthetic hourly rows to {SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()
