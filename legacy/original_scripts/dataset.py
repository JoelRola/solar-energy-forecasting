import pandas as pd
from pathlib import Path
import os

# 1. First verify we can find the input file
try:
    input_path = Path('data/hourlydailyenergy.csv')
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path.absolute()}")

    # loading the data with proper column handling
    energy_df =  pd.read_csv(
         input_path,
           parse_dates=['Date & Time'],
        dayfirst=True,
          skipinitialspace=True
    )
    print("✅ Data loaded successfully")
except Exception as e:
    print(f"❌ Failed to load data: {str(e)}")
    exit()

# cleaning and preparing the data
energy_df.columns =  energy_df.columns.str.strip()
column_mapping = {
      'Date & Time': 'date',
    'Daily Supply': 'daily_supply',
     'Daily Demand': 'daily_demand',
    'Daily Mismatch': 'daily_mismatch'
}
energy_df.rename(columns={
    k: v for k, v in column_mapping.items()
     if k in energy_df.columns
},     inplace=True)

# creating an hourly time series for the year of 2021
try:
    hourly_dates = pd.date_range(
        start='2021-01-01',
        end='2021-12-31 23:00:00',
        freq='h'
    )
    hourly_energy = pd.DataFrame({'datetime': hourly_dates})
    hourly_energy['date'] = hourly_energy['datetime'].dt.normalize()
    hourly_energy = pd.merge(hourly_energy, energy_df, on='date', how='left')
except Exception as e:
    print(f"❌ Failed to create time series: {str(e)}")
    exit()

# 5. Handle output directory creation
try:
    # Try multiple possible output locations
    output_locations = [
        Path.home() / 'Desktop',
        Path.home() / 'Documents',
        Path.cwd() / 'output',
        Path(os.getenv('TEMP', ''))
    ]

    for location in output_locations:
        try:
            location.mkdir(exist_ok=True, parents=True)
            output_path = location / 'hourly_energy_data.csv'

            # Test write permissions
            test_file = location / 'write_test.txt'
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)

            # If we get here, this location works
            break
        except:
            continue
    else:
        raise OSError("Could not find a writable output directory")

    # 6. Save the data
    hourly_energy.to_csv(output_path, index=False)
    print(f"✅ Success! File saved to: {output_path.absolute()}")
    print("\nFile info:")
    print(f"Size: {output_path.stat().st_size/1024:.1f} KB")
    print(f"Created: {pd.to_datetime(output_path.stat().st_ctime)}")

except Exception as e:
    print(f"❌ Failed to save file: {str(e)}")
    print("\nAttempting alternative save methods...")

    # Fallback 1: Save to current directory
    try:
        output_path = Path.cwd() / 'hourly_energy_data.csv'
        hourly_energy.to_csv(output_path, index=False)
        print(f"✅ Saved to current directory: {output_path.absolute()}")
    except Exception as e:
        print(f"❌ Fallback 1 failed: {str(e)}")

        # Fallback 2: Save as pickle
        try:
            output_path = Path.cwd() / 'hourly_energy_data.pkl'
            hourly_energy.to_pickle(output_path)
            print(f"✅ Saved as pickle file: {output_path.absolute()}")
        except Exception as e:
            print(f"❌ All save attempts failed. Final error: {str(e)}")
            print("\nAvailable columns:", hourly_energy.columns.tolist())
            print("First 3 rows:")
            print(hourly_energy.head(3))

import pandas as pd
from pathlib import Path
import os

# 1. Get the CORRECT project root path (fixes double .venv issue)
project_root = Path(__file__).parent
print(f"Project root: {project_root}")

# 2. Define the correct data path options
possible_data_locations = [
    project_root / 'data',                          # ./data
    project_root / '.venv' / 'data',               # ./.venv/data
    project_root / 'Solar panel dataset and models' / 'data',  # Handles spaces in dir name
    Path.home() / 'Desktop'                        # Fallback to desktop
]

# 3. Search for the file
file_found = False
for data_path in possible_data_locations:
    file_path = data_path / 'hourly_energy_data.csv'
    if file_path.exists():
        print(f"✅ Found file at: {file_path}")
        file_found = True
        break

if not file_found:
    print("\n❌ File not found. Please check:")
    print(f"1. The file should be in one of these locations:")
    for loc in possible_data_locations:
        print(f"   - {loc}")
    print("\n2. Try moving your file to:")
    print(f"   {project_root / 'data'}")

    # Create data directory if it doesn't exist
    (project_root / 'data').mkdir(exist_ok=True)
    print(f"\nCreated 'data' folder at: {project_root / 'data'}")
    print("Please place your CSV file there and rerun the script.")
    exit()

# 4. Load and verify the file
try:
    df = pd.read_csv(file_path)
    print("\nFile loaded successfully!")
    print(f"Columns: {list(df.columns)}")

    # Save a verified copy to project root
    backup_path = project_root / 'ENERGY_DATA_BACKUP.csv'
    df.to_csv(backup_path, index=False)
    print(f"\nBackup saved to: {backup_path}")

except Exception as e:
    print(f"\n❌ Error loading file: {str(e)}")
    print("\nTry these fixes:")
    print("1. Open the file in Notepad to check its contents")
    print("2. Right-click → Properties → Unblock (Windows)")
    print("3. Save a fresh copy from your source")

    import pandas as pd
from pathlib import Path

# Load your data (using the correct path we established)
file_path = Path(__file__).parent / 'data' / 'hourly_energy_data.csv'
df = pd.read_csv(file_path)

# handing duplicate data columns
# keeping only one date column (the more precise datetime)
df = df.drop(columns=['date'], errors='ignore')  # Remove the date-only column
df.rename(columns={'datetime': 'timestamp'}, inplace=True)  # Rename to be clearer

# cleaning weekly data
# identifying weekly columns (they repeat every 168 hours = 7 days * 24 hours)
weekly_cols = [col for col in df.columns if 'Weekly' in col or 'Unnamed' in col]

if weekly_cols:
    print(f"Found weekly/repeating columns: {weekly_cols}")

    # option 1: Keep only the first occurrence each week
    df_weekly = df.iloc[::168, :][weekly_cols]  # Every 168th row
    df = df.drop(columns=weekly_cols)

    # option 2: Or just remove them entirely
    # df = df.drop(columns=weekly_cols)

    print("Weekly data extracted to separate dataframe")

# handling unnamed columns
unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
if unnamed_cols:
    print(f"Removing unnamed columns: {unnamed_cols}")
    df = df.drop(columns=unnamed_cols)

# cleaning remaining data
# convert to proper datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# calculate actual hourly values (instead of evenly distributed)
try:
    df['hourly_supply'] = df['daily_supply'] / 24
    df['hourly_demand'] = df['daily_demand'] / 24
except KeyError:
    print("Couldn't calculate hourly values - column names may differ")

# 5. Save cleaned data
clean_path = Path.home() / 'Desktop' / 'CLEAN_ENERGY_DATA.csv'
df.to_csv(clean_path, index=False)
print(f"\n✅ Cleaned data saved to: {clean_path}")
print("\nFinal columns:", df.columns.tolist())
print("\nSample data:")
print(df.head(3))
