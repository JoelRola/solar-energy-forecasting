import pandas as pd  # <-- Add this import

# Your existing code
model_comparison = pd.DataFrame({
    'Model': ['LSTM', 'Random Forest', 'CNN'],
    'Test R²': [0.89, 0.88, 0.87],
    'Test MAE': [17.93, 18.94, 19.45],
    'Test MAPE (%)': [4.52, 4.87, 5.12]
})
model_comparison.to_csv('model_comparison.csv', index=False)