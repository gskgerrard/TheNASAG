"""Estimate FRP for rows in a VIIRS FIRMS CSV using the trained model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "J1_VIIRS_C2_Global_48h.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "risk_model.keras"


def run_inference(input_path=DATA_PATH, output_path=None):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else OUTPUT_DIR / "future_risk.csv"
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Trained model not found. Run train_model.py first.")
    metadata_path = OUTPUT_DIR / "model_metadata.pkl"
    if not metadata_path.is_file():
        raise FileNotFoundError("Model metadata not found. Retrain with train_model.py.")

    raw = pd.read_csv(input_path)
    metadata = joblib.load(metadata_path)
    aliases = {
        "latitude": "lat", "longitude": "lon",
        "bright_ti4": "brightness_temp", "bright_ti5": "brightness_temp_5",
    }
    normalized = raw.rename(columns={source: target for source, target in aliases.items()
                                     if source in raw.columns and target not in raw.columns})
    missing = [name for name in metadata["features"] if name not in normalized.columns]
    if missing:
        raise ValueError(f"Input CSV is missing model features: {missing}")
    x = metadata_preprocessor().transform(normalized[metadata["features"]])
    predicted_log_frp = tf.keras.models.load_model(MODEL_PATH).predict(x, verbose=0).reshape(-1)
    predicted_frp = np.maximum(0, np.expm1(predicted_log_frp))
    raw["estimated_frp"] = predicted_frp
    # A bounded display score for map coloring; it is not a probability or forecast.
    raw["latest_risk"] = predicted_frp / (predicted_frp + 10.0)
    if "region" not in raw.columns:
        raw["region"] = (
            normalized["lat"].round(2).astype("string") + "," +
            normalized["lon"].round(2).astype("string")
        )
    if "date" not in raw.columns and "acq_date" in raw.columns:
        raw["date"] = raw["acq_date"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)
    print(f"Estimated fire intensity for {len(raw):,} detections; saved {output_path}")
    return raw


def metadata_preprocessor():
    return joblib.load(OUTPUT_DIR / "scaler.pkl")


if __name__ == "__main__":
    run_inference()
