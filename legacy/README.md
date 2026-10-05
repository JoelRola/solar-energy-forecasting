# Recovered historical source

These nine scripts were copied from the original Intelligent Systems coursework directory, where
they had been stored inside its `.venv` folder. They are preserved as provenance snapshots; they
are not the maintained entry points for this repository and may depend on the original working
directory, private datasets, or historical package behavior.

Only filenames were normalized to lowercase portable names. File contents and implementation
logic were copied unchanged:

| Recovered filename | Preserved filename |
|---|---|
| `dataset.py` | `dataset.py` |
| `interpolate_energy_data.py` | `interpolate_energy_data.py` |
| `Random forest.py` | `random_forest.py` |
| `Random forest XGboost.py` | `random_forest_xgboost.py` |
| `lstm_energy.py` | `lstm_energy.py` |
| `LSTM-CNN.py` | `lstm_cnn.py` |
| `CNN.py` | `cnn.py` |
| `BLSTM.py` | `blstm.py` |
| `Comparing models.py` | `comparing_models.py` |

The originals include historical bugs and machine-dependent paths. They are retained for review,
not silently repaired here. The clean implementation lives under `src/solar_forecasting/`.
