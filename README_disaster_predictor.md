# Disaster Risk Prediction System

This project analyzes NASA-style environmental CSV/CVV files, detects significant changes, and estimates disaster risk using a DNN.

## Input

Put your data files here:

    data/

Supported extensions:

    .csv
    .cvv
    .txt

Required columns:

    date
    one or more numeric environmental variables

Optional columns:

    region
    lat
    lon
    label / disaster_label / event / target

If no label column is present, proxy labels are created from statistical extremes.

## Run

    python full_disaster_predictor.py

## Outputs

    output/trend_report.csv
    output/trends.png
    output/trend_map_*.png
    output/future_risk.csv
    output/future_risk.png
    output/risk_model.pkl
    output/scaler.pkl

## Dashboard

Optional Streamlit dashboard:

    pip install streamlit
    streamlit run dashboard.py

### Live NASA FIRMS fire detections

The dashboard can fetch recent global VIIRS fire detections directly from FIRMS.
Set your MAP_KEY in the terminal before starting Streamlit so it is not saved in
the source code or typed into the dashboard each time.

PowerShell:

    $env:NASA_FIRMS_MAP_KEY = "your_map_key"
    streamlit run dashboard.py

You can also enter the key in the dashboard's password field; that value is only
used for the current session. Choose the satellite feed and one to ten lookback
days in the sidebar. The dashboard refreshes its cached FIRMS results every ten
minutes.

The current Fire layer uses real FIRMS detections. Local CO, CO₂, smoke, and haze
layers still depend on the local files and derived indicators; they are not live
gas concentration measurements.
