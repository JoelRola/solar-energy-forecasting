# Results and provenance

The original coursework report recorded this ranking: LSTM, BLSTM, Random Forest, LSTM-CNN,
Random Forest + XGBoost, CNN. It is a historical report finding, not a result imposed on evaluation.

Recovered metric artifacts disagree in scale and coverage. The legacy LSTM comparison CSV records
LSTM ahead of Random Forest and CNN, but separate BLSTM and LSTM-CNN metric files report much
smaller test errors. No saved XGBoost comparison metrics were recovered. The scripts, metric files,
and saved LSTM checkpoint also do not form one consistent experiment snapshot.

| Historical artifact | Train R² / MAE / RMSE / MAPE | Test R² / MAE / RMSE / MAPE |
|---|---|---|
| LSTM (`model_metrics.csv`) | 0.92 / 14.72 / 19.85 / 3.21% | 0.89 / 17.93 / 23.76 / 4.52% |
| BLSTM (`blstm_model_metrics.csv`) | 0.9964 / 1.4220 / 1.9872 / 1.0115% | 0.9770 / 0.8807 / 1.3072 / 0.8571% |
| LSTM-CNN (`lstm_cnn_hybrid_metrics.csv`) | 0.9960 / 1.2626 / 2.0976 / 0.9409% | 0.9598 / 1.0946 / 1.7283 / 1.1139% |

The legacy `model_comparison.csv` contains test rows for LSTM (0.89 R², 17.93 MAE, 4.52% MAPE),
Random Forest (0.88, 18.94, 4.87%), and CNN (0.87, 19.45, 5.12%); it does not contain RMSE or
train metrics. These are the original saved values, including their original CSV headers.

New runs append rows to `results/reproduced/metrics/model_comparison.csv` and save figures under
`results/reproduced/figures/`. These are not represented as matching the report unless a future
reproduction actually establishes that.
