import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os

def find_data_file(filename):
    """Search for data file in multiple locations"""
    search_paths = [
        Path(__file__).parent / 'data',               # ./data
        Path(__file__).parent,                        # Current directory
        Path.home() / 'Desktop',                      # Desktop
        Path(__file__).parent / '..' / 'data',        # ../data
        Path('C:/Temp')                               # System temp
    ]

    for path in search_paths:
        full_path = path / filename
        if full_path.exists():
            print(f"✅ Found file at: {full_path}")
            return full_path

    print("\n❌ File not found in these locations:")
    for path in search_paths:
        print(f" - {path}")
    return None

def load_data():
    """Load the cleaned energy dataset"""
    data_file = 'CLEAN_ENERGY_DATA.csv'
    data_path = find_data_file(data_file)

    if not data_path:
        alternatives = ['hourly_energy_data.csv', 'ENERGY_DATA_BACKUP.csv']
        for alt in alternatives:
            print(f"\nTrying alternative: {alt}")
            data_path = find_data_file(alt)
            if data_path:
                break

    if not data_path:
        raise FileNotFoundError(
            f"Could not find data file. Please ensure '{data_file}' exists in: "
            f"{Path(__file__).parent / 'data'}"
        )

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    return df

def interpolate_missing(df):
    """Apply linear interpolation to key columns"""
    cols_to_interpolate = [col for col in ['hourly_supply', 'hourly_demand', 'hourly_mismatch']
                           if col in df.columns]

    if not cols_to_interpolate:
        available = [col for col in df.columns if not col.startswith('Unnamed')]
        raise KeyError(
            f"No valid columns found to interpolate. Available columns:\n{available}"
        )

    for col in cols_to_interpolate:
        missing_before = df[col].isna().sum()
        df[f'{col}_interpolated'] = df[col].interpolate(method='linear')
        missing_after = df[f'{col}_interpolated'].isna().sum()
        print(f"- {col}: Filled {missing_before - missing_after} missing values")

    return df

def main():
    print("\n=== Energy Data Interpolation Tool ===")
    try:
        df = load_data()
        print("\nOriginal data summary:")
        print(df.info())

        df = interpolate_missing(df)

        # Save results
        output_path = Path(__file__).parent / 'data' / 'INTERPOLATED_ENERGY_DATA.csv'
        output_path.parent.mkdir(exist_ok=True)  # Ensure directory exists
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved interpolated data to: {output_path}")

        # Verify save
        if output_path.exists():
            print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
            print("Process completed successfully!")
        else:
            print("⚠️ Warning: File save verification failed")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your file is named 'CLEAN_ENERGY_DATA.csv'")
        print("2. Place it in a 'data' folder next to this script")
        print("3. Verify the file contains 'timestamp' column")

if __name__ == "__main__":
    main()