import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# ------------------------------
# 1. Load and Preprocess Data (Same as LSTM)
# ------------------------------
try:
    df = pd.read_csv('data/INTERPOLATED_ENERGY_DATA.csv', parse_dates=['timestamp'])
    print("✅ Data loaded successfully. Columns:", df.columns.tolist())
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Normalize data (optional for Random Forest, but kept for consistency)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['hourly_supply_interpolated', 'hourly_demand_interpolated']])

# ------------------------------
# create Time-Series Windows for Random Forest
# ------------------------------
def create_dataset(data, lookback=24):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        # Flatten the 24-hour window into a single row (2D input)
        X.append(data[i:(i+lookback), :].flatten())
        y.append(data[i + lookback, 1])  # Predict next hour's demand
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ------------------------------
# train random forest modell
# ------------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Prevent overfitting
    random_state=42    # Reproducibility
)
rf_model.fit(X_train, y_train)

# ------------------------------
# evaluation metrics
# ------------------------------
def inverse_transform_demand(scaler, X_flat, y_pred):
    """Convert scaled predictions back to original units (MWh)."""
    # Reshape X_flat to original window shape (n_samples, lookback, n_features)
    X_reshaped = X_flat.reshape(-1, 24, 2)
    # Use the last hour's supply feature as a dummy for inverse scaling
    dummy = np.zeros((len(y_pred), 2))
    dummy[:, 0] = X_reshaped[:, -1, 0]  # Last hour's supply
    dummy[:, 1] = y_pred                # Predicted demand
    return scaler.inverse_transform(dummy)[:, 1]

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

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

print_metrics(y_train_actual, y_train_pred, "Random Forest - Training Set")
print_metrics(y_test_actual, y_test_pred, "Random Forest - Test Set")

# ------------------------------
# 5. Visualizations (Same as LSTM)
# ------------------------------
plt.figure(figsize=(15, 5))

# Plot Actual vs. Predicted (Test Set)
plt.subplot(1, 2, 1)
plt.plot(y_test_actual[:100], label='Actual', color='blue', alpha=0.6)
plt.plot(y_test_pred[:100], label='Predicted', color='green', linestyle='--')
plt.title("Random Forest: Actual vs. Predicted Demand")
plt.xlabel("Time (Hours)")
plt.ylabel("Demand (MWh)")
plt.legend()

# Feature Importance
plt.subplot(1, 2, 2)
importances = rf_model.feature_importances_
# Reshape importances to match window structure
importances_reshaped = importances.reshape(24, 2)  # (lookback, n_features)
plt.bar(range(importances_reshaped.shape[0]), importances_reshaped[:, 1], color='orange')
plt.title("Feature Importance (Demand Prediction)")
plt.xlabel("Hours Before Prediction")
plt.ylabel("Importance Score")

plt.tight_layout()
plt.show()