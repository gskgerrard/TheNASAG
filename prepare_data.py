import pandas as pd
import glob
import os

# Define paths
data_dir = "data"
output_file = "data/historical_hazards.csv"

# Find all .cvv files in the data folder
files = glob.glob(os.path.join(data_dir, "*.cvv"))

all_data = []

print(f"Found {len(files)} data files. Combining now...")

for file_path in files:
    try:
        # Reading the .cvv files
        df = pd.read_csv(file_path)
        
        # Basic cleaning: ensure date is recognized
        date_cols = ["date", "time", "datetime", "timestamp", "acq_date", "acquisition_date"]
        for col in date_cols:
            if col in df.columns:
                df = df.rename(columns={col: "date"})
                break
        
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        
        # Drop rows where date is missing
        df = df.dropna(subset=["date"])
        
        all_data.append(df)
        print(f"Successfully loaded: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if all_data:
    # Combine all files into one
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure we have the columns the model needs
    # If they don't exist, we create them as 0
    needed_cols = ['fire_intensity', 'heat_intensity', 'smoke_intensity', 
                     'haze_intensity', 'flood_intensity', 'co2_intensity', 'risk_score']
    for col in needed_cols:
        if col not in master_df.columns:
            master_df[col] = 0.0

    # Save the master file
    master_df.to_csv(output_file, index=False)
    print(f"--- SUCCESS: Master file created at {output_file} ---")
else:
    print("No files were found to combine.")
