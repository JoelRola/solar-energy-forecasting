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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# 1. Load and Preprocess Data (Same as LSTM/RF)
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
# create time series windows the same way as LSTM
# ------------------------------
def create_dataset(data, lookback=24):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        X.append(data[i:(i+lookback), :])  # 24-hour window with both features
        y.append(data[i + lookback, 1])    # Predict next hour's demand
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# reshape for CNN (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 24, 2)
X_test = X_test.reshape(X_test.shape[0], 24, 2)

# ------------------------------
# build and Train CNN Model
# ------------------------------
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(24, 2)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# training with early stopping
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# ------------------------------
# evaluate Metrics (same as LSTM)
# ------------------------------
def inverse_transform_demand(scaler, X, y_pred):
    """Convert scaled predictions back to original units (MWh)."""""
    dummy = np.zeros((len(y_pred), X.shape[2]))
    dummy[:, 1] = y_pred.flatten()  # Demand is at index 1
    return scaler.inverse_transform(dummy)[:, 1]

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Inverse scaling
y_train_actual = inverse_transform_demand(scaler, X_train, y_train)
y_train_pred = inverse_transform_demand(scaler, X_train, y_pred_train)
y_test_actual = inverse_transform_demand(scaler, X_test, y_test)
y_test_pred = inverse_transform_demand(scaler, X_test, y_pred_test)

# Calculate metrics
def print_metrics(y_true, y_pred, label):
    print(f"\n📊 **{label} Metrics**")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f} MWh")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} MWh")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}%")

print_metrics(y_train_actual, y_train_pred, "CNN - Training Set")
print_metrics(y_test_actual, y_test_pred, "CNN - Test Set")

# ------------------------------
# 5. Visualizations (Same as LSTM/RF)
# ------------------------------
plt.figure(figsize=(15, 5))

# Plot Actual vs. Predicted (Test Set)
plt.subplot(1, 2, 1)
plt.plot(y_test_actual[:100], label='Actual', color='blue', alpha=0.6)
plt.plot(y_test_pred[:100], label='Predicted', color='purple', linestyle='--')
plt.title("CNN: Actual vs. Predicted Demand")
plt.xlabel("Time (Hours)")
plt.ylabel("Demand (MWh)")
plt.legend()

# Plot Training History
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("CNN Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()

plt.tight_layout()
plt.show()