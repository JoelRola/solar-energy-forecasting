import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

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
# create feauture windows similarly to other models
# ------------------------------
def create_dataset(data, lookback=24):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        X.append(data[i:(i+lookback), :].flatten())  # Flatten the window
        y.append(data[i + lookback, 1])              # Predict next hour's demand
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ------------------------------
# train random forest
# ------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ------------------------------
# train XGBOOST
# ------------------------------
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# ------------------------------
# both models evaluation
# ------------------------------
def inverse_transform_demand(scaler, y_pred):
    """Convert scaled predictions back to original units (MWh)."""
    dummy = np.zeros((len(y_pred), 2))
    dummy[:, 1] = y_pred.flatten()
    return scaler.inverse_transform(dummy)[:, 1]

def evaluate_model(model, X_train, X_test, model_name):
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Inverse scaling
    y_train_actual = inverse_transform_demand(scaler, y_train)
    y_train_pred = inverse_transform_demand(scaler, y_pred_train)
    y_test_actual = inverse_transform_demand(scaler, y_test)
    y_test_pred = inverse_transform_demand(scaler, y_pred_test)

    # Calculate metrics
    def print_metrics(y_true, y_pred, label):
        print(f"\n📊 **{model_name} {label} Metrics**")
        print(f"R²: {r2_score(y_true, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f} MWh")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} MWh")
        print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred) * 100:.2f}%")

    print_metrics(y_train_actual, y_train_pred, "Training Set")
    print_metrics(y_test_actual, y_test_pred, "Test Set")

    return y_test_actual, y_test_pred

# Evaluate both models
rf_actual, rf_pred = evaluate_model(rf_model, X_train, X_test, "Random Forest")
xgb_actual, xgb_pred = evaluate_model(xgb_model, X_train, X_test, "XGBoost")

# ------------------------------
# 6. Enhanced Visualizations
# ------------------------------
plt.figure(figsize=(18, 12))

# Plot Actual vs Predicted (First 100 hours)
plt.subplot(2, 2, 1)
plt.plot(rf_actual[:100], label='Actual', color='black', alpha=0.6)
plt.plot(rf_pred[:100], label='Random Forest', color='green', linestyle='--')
plt.plot(xgb_pred[:100], label='XGBoost', color='blue', linestyle='-.')
plt.title("Test Set: Model Predictions (First 100 Hours)")
plt.xlabel("Time (Hours)")
plt.ylabel("Demand (MWh)")
plt.legend()

# Residuals Comparison
plt.subplot(2, 2, 2)
plt.scatter(rf_pred, rf_actual - rf_pred, alpha=0.5, color='green', label='Random Forest')
plt.scatter(xgb_pred, xgb_actual - xgb_pred, alpha=0.5, color='blue', label='XGBoost')
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals Comparison")
plt.xlabel("Predicted Demand (MWh)")
plt.ylabel("Residuals")
plt.legend()

# Error Distribution
plt.subplot(2, 2, 3)
plt.hist(rf_actual - rf_pred, bins=30, color='green', alpha=0.5, edgecolor='black', label='Random Forest')
plt.hist(xgb_actual - xgb_pred, bins=30, color='blue', alpha=0.5, edgecolor='black', label='XGBoost')
plt.title("Error Distribution")
plt.xlabel("Prediction Error (MWh)")
plt.ylabel("Frequency")
plt.legend()

# Feature Importance (XGBoost)
plt.subplot(2, 2, 4)
feature_importance = xgb_model.feature_importances_
plt.bar(range(len(feature_importance)), feature_importance, color='blue')
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")

plt.tight_layout()
plt.savefig('tree_model_results.png', dpi=300)
plt.show()

# ------------------------------
# 7. Save Metrics to CSV
# ------------------------------
def get_metrics(y_true, y_pred):
    return {
        'R²': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE (%)': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'XGBoost', 'XGBoost'],
    'Dataset': ['Train', 'Test', 'Train', 'Test'],
    **{k: [get_metrics(y_train_actual, inverse_transform_demand(scaler, rf_model.predict(X_train)))[k],
           get_metrics(rf_actual, rf_pred)[k],
           get_metrics(y_train_actual, inverse_transform_demand(scaler, xgb_model.predict(X_train)))[k],
           get_metrics(xgb_actual, xgb_pred)[k]] for k in ['R²', 'MAE', 'RMSE', 'MAPE (%)']}
})

metrics_df.to_csv('tree_model_metrics.csv', index=False)
print("\n📈 Metrics saved to 'tree_model_metrics.csv'")
print("🖼️ Visualizations saved to 'tree_model_results.png'")