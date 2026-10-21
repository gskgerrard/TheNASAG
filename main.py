import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from data_loader import generate_synthetic_nasa_data, prepare_sequences

# --- CONFIGURATION ---
WINDOW_SIZE = 10
LAT_SIZE = 64
LON_SIZE = 64
CHANNELS = 2 # 0: CO, 1: IR

# 1. LOAD DATA
print("--- Loading Data ---")
raw_data = generate_synthetic_nasa_data(timesteps=60, lat=LAT_SIZE, lon=LON_SIZE)
X, y = prepare_sequences(raw_data, window_size=WINDOW_SIZE)

# Split into Train/Test (Last 10 time steps for testing)
X_train, X_test = X[:-10], X[-10:]
y_train, y_test = y[:-10], y[-10:]

# 2. BUILD THE DNN (ConvLSTM)
# This model 'sees' the earth like a video of infrared heat maps
def build_complete_model():
    model = models.Sequential([
        # ConvLSTM captures the "Where" (spatial) and "When" (temporal)
        layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), 
                           padding='same', return_sequences=True, 
                           input_shape=(WINDOW_SIZE, LAT_SIZE, LON_SIZE, CHANNELS)),
        layers.BatchNormalization(),
        
        layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), 
                           padding='same', return_sequences=False),
        layers.BatchNormalization(),
        
        # Output layer: predicting the next frame of CO and IR
        layers.Conv2D(filters=CHANNELS, kernel_size=(3, 3), 
                       activation='sigmoid', padding='same')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

print("--- Building Model ---")
model = build_complete_model()
model.summary()

# 3. TRAIN THE MODEL
print("\n--- Training Model ---")
model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# 4. PREDICT AND VISUALIZE (THE INFRARED OUTPUT)
print("\n--- Generating Results ---")
predictions = model.predict(X_test)

# Visualization: Let's look at the very last prediction
# We will display the CO Concentration as a "Heatmap" (Infrared Style)
plt.figure(figsize=(12, 5))

# Actual CO (Channel 0)
plt.subplot(1, 2, 1)
plt.title("Actual CO (Infrared Style)")
plt.imshow(y_test[-1, :, :, 0], cmap='inferno') # 'inferno' looks like heat
plt.colorbar(label='CO Density')

# Predicted CO (Channel 0)
plt.subplot(1, 2, 2)
plt.title("DNN Predicted CO Change")
plt.imshow(predictions[-1, :, :, 0], cmap='inferno')
plt.colorbar(label='Predicted CO Density')

plt.tight_layout()
plt.show()

print("SUCCESS: Model has been trained and visualization generated.")
