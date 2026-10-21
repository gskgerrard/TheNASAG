import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

def run_inference():
    # Load saved model and scaler
    model = tf.keras.models.load_model('output/risk_model.h5')
    scaler = joblib.load('output/scaler.pkl')

    # Load new data (e.g., the global dataset)
    df = pd.read_csv('J1_VIIRS_C2_Global_48h.csv')
    
    # Prepare features
    features = ['lat', 'lon', 'brightness_temp', 'precip_mm_day', 'soil_moisture']
    X = df[features].fillna(df[features].mean())
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)
    
    # Create a risk score (0 to 1)
    df['fire_risk_score'] = (predictions > 0).astype(int) 
    
    # Save results
    df.to_csv('output/future_risk.csv', index=False)
    print("Prediction complete. Results saved to output/future_risk.csv")

if __name__ == "__main__":
    run_inference()
