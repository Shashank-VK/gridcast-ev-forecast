# GridCast — EV Fleet Growth Forecaster

## Overview
Green Grid AI ranked Bengaluru zones for EV charging sites from a static snapshot
(210K+ registration records). GridCast adds the missing time dimension: it forecasts
how the registered EV fleet in each zone grows month by month, so charging-site
planning can use projected demand instead of today's snapshot.

## Problem
A viability score built on current registrations goes stale within months because
EV fleets grow at different rates per zone. Planners need a per-zone forecast with
an honestly measured error, not a single growth guess.

## What it does
1. Loads a real public dataset: monthly registered EV population by county
   (Washington State open data, data.wa.gov dataset 3d5d-sdqb, "Electric Vehicle
   Population Size History By County"), vendored at `data/ev_population_by_county_raw.csv`
   with provenance in `data/PROVENANCE.md`. Counties act as zones.
2. Cleans the data into one monthly time series per zone (passenger vehicles, WA).
3. Runs a stationarity check (Augmented Dickey-Fuller, statsmodels) per zone and
   records the result.
4. Builds lag / rolling-mean / calendar features and trains a scikit-learn
   GradientBoostingRegressor per zone on a chronological train split.
5. Evaluates on the held-out last 12 months against a seasonal-naive baseline
   (same month last year) and reports MAPE and RMSE per zone and overall.
6. Writes `outputs/metrics.json` (measured metrics + ADF results + data range)
   and `outputs/forecast.csv` (next-6-month forecast for the 5 largest zones).

## Non-goals
- No web UI, no API server, no deep learning, no PySpark/Databricks.
- No claim about Bengaluru data: the dataset is Washington State open data and
  the README says so plainly. The method transfers; the numbers are WA numbers.
- No synthetic or generated data anywhere in the pipeline.

## Tech stack
Python 3.12, pandas, numpy, scikit-learn, statsmodels, pytest. CLI via
`python -m gridcast run`. Virtualenv in `.venv`, deps in `requirements.txt`.

## Verification
- `pytest tests/ -q` passes.
- `.builder-os/init.sh` exits 0 (install + tests + a real pipeline run).
- `outputs/metrics.json` from the real run is the only source of numbers quoted
  in the README.
