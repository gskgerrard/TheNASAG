import os
import numpy as np
import pandas as pd

# Create the data folder
os.makedirs("data", exist_ok=True)

np.random.seed(42)

# 5 years of daily data
dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
n_days = len(dates)

t = np.arange(n_days, dtype=float)
season = np.sin(2.0 * np.pi * t / 365.25)


def make_region_file(name, lat, lon, region_type):
    if region_type == "wildfire":
        # California-like: rising temperature, drying soil
        precip = np.maximum(
            0.0,
            2.0 + 3.0 * season + np.random.exponential(1.5, n_days) - 0.001 * t
        )

        tmax = (
            18.0
            + 8.0 * season
            + 0.015 * t
            + np.random.normal(0.0, 2.5, n_days)
        )

        soil = np.clip(
            0.25
            - 0.0001 * t
            + 0.05 * season
            + np.random.normal(0.0, 0.02, n_days),
            0.05,
            0.90
        )

        ndvi = np.clip(
            0.40
            - 0.00005 * t
            + 0.10 * season
            + np.random.normal(0.0, 0.03, n_days),
            0.10,
            0.90
        )

        # Wildfire disaster proxy: extreme heat + dry soil
        disaster_label = ((tmax > 35.0) & (soil < 0.15)).astype(int)

    elif region_type == "flood":
        # Bangladesh-like: increasing heavy rainfall
        precip = np.maximum(
            0.0,
            8.0 + 12.0 * season + 0.008 * t + np.random.exponential(4.0, n_days)
        )

        tmax = (
            28.0
            + 4.0 * season
            + np.random.normal(0.0, 1.5, n_days)
        )

        soil = np.clip(
            0.45
            + 0.0001 * t
            + 0.10 * season
            + np.random.normal(0.0, 0.05, n_days),
            0.05,
            0.95
        )

        ndvi = np.clip(
            0.65
            + 0.05 * season
            + np.random.normal(0.0, 0.04, n_days),
            0.10,
            0.95
        )

        # Flood disaster proxy: extreme rainfall
        disaster_label = (precip > 45.0).astype(int)

    else:
        # Amazon-like: drought / vegetation stress
        precip = np.maximum(
            0.0,
            5.0 + 4.0 * season - 0.003 * t + np.random.exponential(2.0, n_days)
        )

        tmax = (
            26.0
            + 2.0 * season
            + 0.005 * t
            + np.random.normal(0.0, 1.2, n_days)
        )

        soil = np.clip(
            0.35
            - 0.00015 * t
            + 0.05 * season
            + np.random.normal(0.0, 0.03, n_days),
            0.05,
            0.90
        )

        ndvi = np.clip(
            0.75
            - 0.0002 * t
            + 0.05 * season
            + np.random.normal(0.0, 0.04, n_days),
            0.20,
            0.95
        )

        # Drought disaster proxy: dry soil + vegetation stress
        disaster_label = ((soil < 0.18) & (ndvi < 0.45)).astype(int)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "region": name,
            "lat": lat,
            "lon": lon,
            "precip_mm_day": np.round(precip, 2),
            "tmax_C": np.round(tmax, 2),
            "soil_moisture": np.round(soil, 4),
            "ndvi": np.round(ndvi, 4),
            "disaster_label": disaster_label,
        }
    )

    filename = os.path.join("data", f"synthetic_{name}_data.cvv")
    df.to_csv(filename, index=False)

    print(
        f"Generated {filename} | "
        f"Rows: {len(df)} | "
        f"Disaster labels: {int(disaster_label.sum())}"
    )


regions = [
    ("California_Coast", 36.7, -119.4, "wildfire"),
    ("Bangladesh_Delta", 23.6, 90.3, "flood"),
    ("Amazon_Basin", -3.4, -62.2, "drought"),
]

for name, lat, lon, region_type in regions:
    make_region_file(name, lat, lon, region_type)

print("\nDone. Synthetic example files are inside the data/ folder; these are not NASA observations.")
