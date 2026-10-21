import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# CONFIGURATION (Adjust these settings!)
# ============================================================
DATA_PATH = "data"                # Folder containing your .cvv or .csv files
OUTPUT_DIR = "outputs"            # Where results will be saved
WINDOW_SIZE = 30                  # Look back 30 days/steps
HORIZON = 1                       # Predict 1 step ahead
TEST_FRACTION = 0.2               # 20% of data for testing
EPOCHS = 50
BATCH_SIZE = 32
LABEL_COLUMN = None               # Set to your target column name (e.g., 'risk_score')
HIGH_RISK_THRESHOLD = 0.8         # For auto-flagging if no label exists
LOW_RISK_THRESHOLD = 0.2

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_and_merge_data(path):
    """Loads all .cvv and .csv files and merges them into one dataframe."""
    all_dfs = []
    files = [f for f in os.listdir(path) if f.endswith(('.cvv', '.csv'))]
    
    if not files:
        print(f"No files found in {path}")
        return None

    for file in files:
        try:
            df = pd.read_csv(os.path.join(path, file))
            # Standardize date column name
            date_cols = ["date", "time", "datetime", "timestamp"]
            for col in date_cols:
                if col in df.columns:
                    df = df.rename(columns={col: "date"})
                    break
            
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            df = df.dropna(subset=["date"])
            all_dfs.append(df)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure required columns exist
    needed_cols = ['fire_intensity', 'heat_intensity', 'smoke_intensity', 
                    'haze_intensity', 'flood_intensity', 'co2_intensity', 'risk_score']
    for col in needed_cols:
        if col not in master_df.columns:
            master_df[col] = 0.0
    
    return master_df.sort_values("date").reset_index(drop=True)

def prepare_sequences(df, feature_cols, label_col, window, horizon):
    """Converts a dataframe into 3D arrays [samples, time_steps, features] for LSTM."""
    X, y = [], []
    data_array = df[feature_cols].values
    label_array = df[label_col].values if label_col else np.zeros(len(df))

    for i in range(len(df) - window - horizon):
        X.append(data_array[i : i + window])
        y.append(label_array[i + window : i + window + horizon])
    
    return np.array(X), np.array(y)

def train_dnn_model(X_train, y_train, X_test, y_test):
    """Builds and trains a Long Short-Term Memory (LSTM) network."""
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid') # Use 'linear' if predicting continuous values
    ])

    model.compile(optimizer='adam', loss='mse') # Change to 'binary_crossentropy' if binary classification
    
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    print("Starting training...")
    model.fit(X_train, y_train, 
               validation_data=(X_test, y_test), 
               epochs=EPOCHS, 
               batch_size=BATCH_SIZE, 
               callbacks=[early_stop],
               verbose=1)
    
    return model

# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    df = load_and_merge_data(DATA_PATH)
    if df is None or df.empty:
        print("No data to process. Check your 'data' folder.")
        return

    # 2. Identify Features and Labels
    # Automatically find numeric columns for features
    feature_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] 
                     and c not in ['date', 'risk_score']]
    
    # Determine target label
    if LABEL_COLUMN and LABEL_COLUMN in df.columns:
        target = LABEL_COLUMN
    else:
        # If no label, use 'risk_score' or create a dummy flag
        target = 'risk_score' if 'risk_score' in df.columns else feature_cols[0]
        print(f"Warning: No label column specified. Using '{target}' as target.")

    # 3. Scaling
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler for future use
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 4. Create Sequences
    X, y = prepare_sequences(df, feature_cols, target, WINDOW_SIZE, HORIZON)
    
    # 5. Split Data (Time Series Split)
    split_idx = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training shape: {X_train.shape}")

    # 6. Train Model
    model = train_dnn_model(X_train, y_train, X_test, y_test)

    # 7. Save Model
    model.save(os.path.join(OUTPUT_DIR, "emergency_prediction_model.h5"))
    print(f"Model saved to {OUTPUT_DIR}")

    # 8. Simple Forecast Example
    last_window = X[-1].reshape(1, WINDOW_SIZE, len(feature_cols))
    prediction = model.predict(last_window)
    print(f"\nPredicted Risk for next period: {prediction[0][0]}")

if __name__ == "__main__":
    main()
