# Methodology

## Forecast target and inputs

Each model uses the preceding 24 hourly rows of interpolated supply and demand to predict demand in
the next row. Tree models flatten the `(24, 2)` window to 48 values; neural models retain the
time-by-feature shape.

## Data transformation

The coursework CSV contains daily totals. The recovered pipeline expands each daily row to 24
timestamps, then divides daily supply and demand by 24. Linear interpolation is applied to available
hourly columns. These values are uniform daily allocations, not measured hourly supply/demand.

## Split and scaling

Windows are split chronologically, with the last 20% reserved for test. To preserve the recovered
method, MinMaxScaler is fitted to the complete time series before window splitting. This allows the
test-period range to influence scaling and is a leakage limitation. Neural models also use the test
portion as validation data, matching the recovered scripts. These choices must be considered when
interpreting metrics.

Windows are generated before the chronological window split, matching the recovered code. As a
result, the final training labels overlap the input period of the first test windows. This is an
additional boundary leakage limitation; it is retained for fidelity and called out rather than
silently changing the original experiment.

The clean sequence generator includes every complete 24-row input plus next-row target. The legacy
loop stopped one iteration early and omitted the final otherwise-valid window; this small sample
count correction is explicit in the clean implementation.

The clean Keras constructors use an explicit `Input` layer to define the same `(24, 2)` input shape
without relying on the deprecated `input_shape` argument on a subsequent layer. The trainable model
layers, order, and hyperparameters remain unchanged.

## Architectures

The clean constructors preserve the recovered LSTM, bidirectional LSTM, CNN, CNN-LSTM, Random
Forest, and XGBoost settings. The Random Forest and XGBoost estimators are fitted independently;
there is no ensemble prediction.

## Evaluation

R², MAE, RMSE, and MAPE are calculated on inverse-scaled demand values. MAPE excludes actual values
whose absolute magnitude is at most `1e-8`, avoiding division by values effectively equal to zero.
If all actual values are near zero, MAPE is recorded as NaN.
