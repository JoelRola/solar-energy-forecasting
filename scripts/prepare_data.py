"""Build processed hourly data from an authorized daily input CSV."""

import argparse

from solar_forecasting.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from solar_forecasting.preprocessing import prepare_energy_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(RAW_DATA_PATH), help="Authorized daily CSV")
    parser.add_argument("--output", default=str(PROCESSED_DATA_PATH), help="Processed CSV path")
    args = parser.parse_args()
    frame = prepare_energy_data(args.input, args.output)
    print(f"Saved {len(frame)} hourly rows to {args.output}")


if __name__ == "__main__":
    main()
