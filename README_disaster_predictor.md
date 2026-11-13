# Earth Emergency Dashboard

This project combines local environmental CSV files, NASA EONET event records,
and NASA FIRMS active fire detections. Live services need internet access. FIRMS
also needs a free MAP_KEY.

## Install and run

From this folder, install `requirements.txt`, then start the dashboard:

```powershell
python -m pip install -r requirements.txt
$env:NASA_FIRMS_MAP_KEY = "your_map_key"
python -m streamlit run dashboard.py
```

Get a FIRMS MAP_KEY from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/).
Alternatively, enter it in the dashboard's password field for the current
session. The key is not saved by the app. EONET is public and does not use this
key.

## Data and predictions

Place dated `.csv` or `.cvv` files in `data/`. The dashboard reads these files
and labels local signals as derived indicators. It does not treat proxies as
measured atmospheric concentrations.

Run `python cvv.py` to generate synthetic example hazard data. Those examples
are illustrative only and are not NASA observations.

The included `J1_VIIRS_C2_Global_48h.csv` is a saved VIIRS FIRMS sample, not a
live or current feed. The model estimates fire radiative power (FRP) for
detections in this file. It does not predict future disasters or estimate a
probability of a fire. To train and run that estimate:

```powershell
python pipeline.py
```

The model and preprocessing files go into `output/`; estimates are written to
`output/future_risk.csv`. The dashboard's **Run Prediction** button runs the
same pipeline. You can also run `python train_model.py` to train only, or
`python run_prediction.py` after training to estimate the bundled sample.

## Live FIRMS feed

The dashboard fetches near-real-time VIIRS or MODIS detections using the
selected source and lookback period. A saved local FIRMS CSV is shown as a
dated snapshot only when the live request is unavailable. FIRMS hotspots are
satellite detections and do not by themselves confirm a wildfire or define a
fire perimeter.
