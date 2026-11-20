# Build brief (Auto Mode — zero questions, decide and build)

Project root: this folder. Read specs.md first; it is the locked spec.
State: .builder-os/feature_list.json (features F1–F4, uphill first),
.builder-os/init.sh is the verification gate.

Build the whole project end to end:

1. Package `src/gridcast/` with modules:
   - `data.py`: load `data/ev_population_by_county_raw.csv`, parse dates,
     coerce counts to int, return a dict of zone -> monthly pandas Series
     (sorted, month-end index, no gaps: reindex to a full monthly range and
     forward-fill only within a zone's observed span). Function
     `train_test_split_series(series, test_months=12)` returning chronological
     (train, test) with train strictly before test.
   - `stationarity.py`: `adf_report(series)` using statsmodels
     `adfuller`, returning dict with statistic, p_value, stationary (p < 0.05).
   - `model.py`: `make_features(series)` building a supervised frame with
     lag_1, lag_2, lag_3, lag_12, roll_mean_3, roll_mean_12, month, and a
     linear time trend; `seasonal_naive_predict(train, periods)` (repeat the
     last 12 observed months); `evaluate_zone(series)` training
     GradientBoostingRegressor on all rows except the last 12 months and
     scoring MAPE + RMSE on the last 12 for both model and baseline.
   - `pipeline.py` + `__main__.py`: `python -m gridcast run` loads data,
     evaluates every zone with at least 48 months of history, computes the
     ADF report per evaluated zone, writes `outputs/metrics.json`
     (keys: data_range, zones_evaluated, overall {model_mape, model_rmse,
     baseline_mape, baseline_rmse}, per_zone, adf) and
     `outputs/forecast.csv` (recursive 6-month-ahead forecast for the 5
     zones with the largest latest EV population; columns zone, month,
     predicted_ev_population).
   Use a src layout: add `pyproject.toml` with package discovery for
   `src/gridcast` and a `gridcast` console entry; tests must import
   `gridcast` (install the project editable in init via requirements or
   conftest sys.path — simplest reliable route: a `conftest.py` at repo
   root adding `src` to sys.path, and `python -m gridcast` run with
   PYTHONPATH=src set inside pipeline invocation in init.sh is NOT allowed
   to change; instead make `__main__.py` work by adding a root-level
   `gridcast` shim is messy — prefer: pyproject + `pip install -e .` in
   requirements step. Add `-e .` handling: put a line in requirements.txt
   is not possible for editable self-install; instead modify nothing in
   init.sh and add `conftest.py` for tests plus a tiny root `gridcast.py`
   is forbidden (shadows package). Cleanest: in `.builder-os/init.sh`
   the install step is fixed, so add `src/gridcast/__init__.py` and set
   PYTHONPATH via a `pytest.ini`/`pyproject.toml` [tool.pytest] pythonpath
   = ["src"], and for `python -m gridcast`, add a root-level `sitecustomize`
   is overkill — simplest correct approach: make the package live at repo
   root as `gridcast/` package directory instead of src layout. DO THAT:
   package directory `gridcast/` at project root, tests at `tests/`,
   data paths resolved relative to the package's parent (project root).
2. Tests first (TDD) in `tests/`: test_data.py (row count > 4000, monthly
   sorted index, split has no overlap and train ends before test begins),
   test_stationarity.py (ADF runs on King county series and returns keys;
   a clearly stationary white-noise series is flagged stationary),
   test_model.py (features have no NaN after warmup; seasonal naive repeats
   last 12 values; evaluate_zone returns finite MAPE/RMSE and beats or
   reports baseline honestly), test_pipeline.py (metrics.json schema after
   a run on the real data: overall keys present, zones_evaluated >= 20).
3. Run `.builder-os/init.sh` until it exits 0. Record evidence (test counts,
   metrics headline numbers) in .builder-os/feature_list.json evidence[]
   and .builder-os/build-log.md, marking each feature passing.
4. README.md at root: problem, method, dataset with provenance link, and the
   REAL measured results copied from outputs/metrics.json (overall model vs
   baseline MAPE/RMSE, zones evaluated, data range). Plain language, no
   hype words, no em dashes.
5. Do not commit anything to git and do not create a GitHub repo (handled
   separately). No secrets anywhere. Keep the root clean: only specs.md,
   README.md, pyproject.toml, requirements.txt, gridcast/, tests/, data/,
   outputs/, .builder-os/ belong here.

Done means: init.sh exit 0, all features passing with evidence recorded.
