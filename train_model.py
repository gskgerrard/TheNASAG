"""Train a model that estimates VIIRS fire radiative power (FRP)."""
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from data_loader import load_and_prepare_data

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "J1_VIIRS_C2_Global_48h.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "risk_model.keras"


def build_model(feature_count):
    model = models.Sequential([
        layers.Input(shape=(feature_count,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="relu"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"FIRMS dataset not found: {DATA_PATH}")
    data = load_and_prepare_data(
        DATA_PATH,
        features=["lat", "lon", "brightness_temp", "scan", "track", "brightness_temp_5"],
        target="frp",
        group_col="acq_date",
        log_target=True,
    )
    x_train, x_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    model = build_model(x_train.shape[1])
    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=50, batch_size=32, callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                                   restore_best_weights=True)
              ], verbose=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(data["preprocessor"], OUTPUT_DIR / "scaler.pkl")
    metadata = {
        "features": data["feature_names"],
        "target": "frp",
        "log_target": True,
        "max_training_frp": float(np.expm1(y_train).max()),
    }
    joblib.dump(metadata, OUTPUT_DIR / "model_metadata.pkl")
    print(f"Model and preprocessing saved under {OUTPUT_DIR}")
    return model


if __name__ == "__main__":
    train()
