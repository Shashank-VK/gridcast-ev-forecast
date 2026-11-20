# Build log — GridCast EV Forecaster

## Decisions made
- Auto Mode (task brief is the PRD; no user questions).
- Dataset: Washington State open data 3d5d-sdqb (monthly EV population by
  county), vendored with provenance — real public data, no synthetic series.
- Model: scikit-learn GradientBoostingRegressor on lag/rolling/calendar
  features vs seasonal-naive baseline; ADF via statsmodels.
- Zones = counties; top 5 by latest EV population get the 6-month forecast.

## Evidence

### F1: Forecasting core
- `pytest tests/test_model.py` passed (3/3 tests passed in 5.93s).
- Verified lag features (lag_1, lag_2, lag_3, lag_12), rolling means (roll_mean_3, roll_mean_12), calendar (month), and linear time trend.
- Warmup period verified: NaNs in first 12 months, zero NaNs for all rows after warmup.
- Evaluated King County holdout (last 12 months):
  - Model MAPE: 4.10%, Model RMSE: 6,119.09
  - Baseline MAPE: 12.01%, Baseline RMSE: 16,979.06

### F2: Data pipeline
- `pytest tests/test_data.py` passed (4/4 tests passed in 4.78s).
- Loaded 39 county zones across 4,602 monthly records (2017-01-31 to 2026-10-31).
- Reindexed and forward-filled within observed spans to guarantee gap-free monthly series.
- Verified chronological train/test split leakage guard: strictly monotonic separation, disjoint indices.

### F3: Stationarity analysis
- `pytest tests/test_stationarity.py` passed (2/2 tests passed in 5.23s).
- Augmented Dickey-Fuller (ADF) test reports statistic, p_value, and stationary bool.
- Verified on King County series (statistic=-2.4414, p_value=0.1304, non-stationary) and stationary white noise.

### F4: CLI run, outputs, and README
- `python -m gridcast run` executed cleanly and exited 0.
- `outputs/metrics.json` generated:
  - Data range: 2017-01-31 to 2026-10-31
  - Zones evaluated: 39 counties
  - Overall Model MAPE: 9.65% vs Baseline MAPE: 19.28%
  - Overall Model RMSE: 468.65 vs Baseline RMSE: 1039.83
- `outputs/forecast.csv` generated:
  - 30 rows (top 5 counties: King, Snohomish, Pierce, Clark, Kitsap across 6 months).
  - Columns: zone, month, predicted_ev_population.
- `README.md` documents problem, method, dataset provenance link, and real measured results without hype words or em dashes.
- Gate `.builder-os/init.sh` exits 0 with 10 passed tests.

## Update 2026-11-20 (post-build, Flash-gate driven)
- Added autocorrelation analysis: `acf_report` in gridcast/stationarity.py
  (statsmodels acf, lag 1 and seasonal lag 12), wired into pipeline metrics
  under `acf` per zone. New test test_acf_report_ar1_and_king.
- Re-verified: init.sh exit 0, 11 tests passed. Overall metrics unchanged
  (MAPE 9.65% vs 19.28% baseline). King ACF lag_1 0.9791, lag_12 0.7026.
