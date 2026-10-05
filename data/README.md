# Data and provenance

The university-supplied dataset may have redistribution restrictions. Do not commit it or derived
copies unless you have explicit permission. The repository expects the authorized daily file at
`data/raw/hourlydailyenergy.csv`; use `scripts/prepare_data.py` to create the processed CSV under
`data/processed/`.

Expected daily source fields are `Date & Time`, `Daily Supply`, `Daily Demand`, and optionally
`Daily Mismatch`. The loader also accepts the normalized names `date`, `daily_supply`,
`daily_demand`, and `daily_mismatch`. Spreadsheet weekly/unnamed columns are discarded during
cleaning. The processed schema includes `timestamp`, daily values, `hourly_supply`,
`hourly_demand`, and interpolated supply/demand features. The historical pipeline divides daily
supply and demand evenly by 24; these are allocations, not observed hourly measurements.

`data/sample/synthetic_energy.csv` is generated synthetic data for documentation and tests. It is
not university data and must never be presented as historical observations or results.

The MIT license in the repository applies to source code only. It does not grant rights to the
university dataset or Global Solar Atlas material.
