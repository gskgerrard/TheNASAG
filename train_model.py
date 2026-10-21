import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import load_and_prepare_data
import joblib
import numpy as np
import json

# Configuration
DATA_PATH = 'data/J1_VIIRS_C2_Global_48h.csv'
MODEL_SAVE_PATH = 'output/risk_model.h5'

def train():
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(DATA_PATH)
    
    # Build the DNN Model
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear') # Predicts Fire Radiative Power (FRP)
    ])

    # Compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train
    print("Starting training...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Save the model and the scaler
    import os
    if not os.path.exists('output'): os.makedirs('output')
    
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, 'output/scaler.pkl')
    print(f"Model saved to {MODEL_SAVE_PATH}")

model.save('earth_emergency_model.h5')
print("Model saved as earth_emergency_model.h5")

# 2. Create a function to predict for new data
def predict_emergency(input_data):
    """
    input_data: A list or numpy array of features 
    (Must have the same number of columns as your training data)
    """
    # Convert input to numpy array and reshape for the model
    data_array = np.array(input_data).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(data_array)
    result_index = np.argmax(prediction)
    
    # Map back to labels (D=Danger, N=Normal)
    labels = ['Normal', 'Danger'] # Adjust based on your specific encoding
    return labels[result_index], prediction[0][result_index]

# --- Example Usage ---
# Replace these numbers with actual real-time sensor data
sample_data = [0.5, 1.2, 0.03, 0.8, 1.1, 0.4, 0.9, 0.2, 0.1, 0.5, 0.6, 0.7] 
result, confidence = predict_emergency(sample_data)
print(f"Prediction: {result} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    train()
