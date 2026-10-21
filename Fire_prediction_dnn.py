from pathlib import Path

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. SETTINGS (Adjust these based on your goal)
# ==========================================
FILE_PATH = Path(__file__).resolve().parent / "data" / "J1_VIIRS_C2_Global_48h.csv"

# Since your header doesn't show an 'emergency' column, 
# replace 'target_column_name' with the actual column you want to predict.
# If you want to predict a category (e.g., 'daynight'), keep it as is.
TARGET_COLUMN = 'daynight' 

# We only want to use these numeric/relevant columns for the "Features"
# I have excluded 'acq_date' and 'acq_time' because they are strings/dates
FEATURE_COLUMNS = [
    'latitude', 'longitude', 'bright_ti4', 'scan', 
    'track', 'confidence', 'bright_ti5', 'frp'
]

def prepare_data():
    df = pd.read_csv(FILE_PATH)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X = pd.get_dummies(X, columns=["confidence"], dtype=np.float32)

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Encoded target classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    print("Data prepared successfully.")

    num_classes = len(np.unique(y_train))
    if num_classes == 2:
        output_layer = Dense(1, activation='sigmoid')
        loss_fn = 'binary_crossentropy'
    else:
        output_layer = Dense(num_classes, activation='softmax')
        loss_fn = 'sparse_categorical_crossentropy'

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        output_layer
    ])

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    print("\nStarting Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
