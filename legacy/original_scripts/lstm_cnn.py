import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# 1. Load and Preprocess Data
# ------------------------------
try:
    df = pd.read_csv('data/INTERPOLATED_ENERGY_DATA.csv', parse_dates=['timestamp'])
    print("✅ Data loaded successfully. Columns:", df.columns.tolist())
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['hourly_supply_interpolated', 'hourly_demand_interpolated']])

# ------------------------------
# create time series windows
# ------------------------------
def create_dataset(data, lookback=24):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        X.append(data[i:(i+lookback), :])  # 24-hour window
        y.append(data[i + lookback, 1])    # Predict next hour's demand
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ------------------------------
# hybrid model construction
# ------------------------------
model = Sequential([
    # CNN Feature Extraction (input shape is 3D: (timesteps, features))
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(24, 2)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # LSTM Temporal Processing
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ------------------------------
# 4. Train the Model
# ------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# ------------------------------
# 5. Evaluate Metrics
# ------------------------------
def inverse_transform_demand(scaler, X, y_pred):
    dummy = np.zeros((len(y_pred), X.shape[2]))
    dummy[:, 1] = y_pred.flatten()
    return scaler.inverse_transform(dummy)[:, 1]

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Inverse scaling
y_train_actual = inverse_transform_demand(scaler, X_train, y_train)
y_train_pred = inverse_transform_demand(scaler, X_train, y_pred_train)
y_test_actual = inverse_transform_demand(scaler, X_test, y_test)
y_test_pred = inverse_transform_demand(scaler, X_test, y_pred_test)

# Metrics
def print_metrics(y_true, y_pred, label):
    print(f"\n📊 {label} Metrics")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f} MWh")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} MWh")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}%")

print_metrics(y_train_actual, y_train_pred, "Training Set")
print_metrics(y_test_actual, y_test_pred, "Test Set")

# ------------------------------
# 6. Visualizations
# ------------------------------
plt.figure(figsize=(15, 10))

# Training History
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()

# Predictions
plt.subplot(2, 2, 2)
plt.plot(y_test_actual[:100], label='Actual', color='blue', alpha=0.6)
plt.plot(y_test_pred[:100], label='Predicted', color='red', linestyle='--')
plt.title("Test Set: Actual vs. Predicted Demand")
plt.xlabel("Time (Hours)")
plt.ylabel("Demand (MWh)")
plt.legend()

# Residuals
plt.subplot(2, 2, 3)
residuals = y_test_actual - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals Plot")
plt.xlabel("Predicted Demand (MWh)")
plt.ylabel("Residuals")

# Error Distribution
plt.subplot(2, 2, 4)
plt.hist(residuals, bins=30, edgecolor='black')
plt.title("Error Distribution")
plt.xlabel("Prediction Error (MWh)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig('lstm_cnn_hybrid_results.png', dpi=300)
plt.show()

# ------------------------------
# 7. Save Metrics
# ------------------------------
metrics_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'R²': [r2_score(y_train_actual, y_train_pred),
           r2_score(y_test_actual, y_test_pred)],
    'MAE': [mean_absolute_error(y_train_actual, y_train_pred),
            mean_absolute_error(y_test_actual, y_test_pred)],
    'RMSE': [np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
             np.sqrt(mean_squared_error(y_test_actual, y_test_pred))],
    'MAPE (%)': [mean_absolute_percentage_error(y_train_actual, y_train_pred) * 100,
                 mean_absolute_percentage_error(y_test_actual, y_test_pred) * 100]
})
metrics_df.to_csv('lstm_cnn_hybrid_metrics.csv', index=False)
