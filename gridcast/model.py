import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from gridcast.data import train_test_split_series


def make_features(series: pd.Series) -> pd.DataFrame:
    """Build supervised frame with lag, rolling-mean, calendar, and trend features.

    Features:
    - lag_1, lag_2, lag_3, lag_12
    - roll_mean_3, roll_mean_12 (using shift(1) to avoid target leakage)
    - month (1-12)
    - trend (linear time trend index)
    """
    df = pd.DataFrame(index=series.index)
    df["lag_1"] = series.shift(1)
    df["lag_2"] = series.shift(2)
    df["lag_3"] = series.shift(3)
    df["lag_12"] = series.shift(12)
    df["roll_mean_3"] = series.shift(1).rolling(window=3).mean()
    df["roll_mean_12"] = series.shift(1).rolling(window=12).mean()
    df["month"] = series.index.month
    df["trend"] = np.arange(len(series), dtype=float)
    return df


def seasonal_naive_predict(train: pd.Series, periods: int = 12) -> np.ndarray:
    """Seasonal naive forecast repeating the last 12 observed months."""
    if len(train) < 12:
        raise ValueError("Series must have at least 12 observations for seasonal naive")
    last_12 = train.iloc[-12:].values
    repeats = int(np.ceil(periods / 12))
    return np.tile(last_12, repeats)[:periods]


def evaluate_zone(series: pd.Series) -> dict[str, float]:
    """Train GradientBoostingRegressor on all rows except last 12 months.

    Scores MAPE and RMSE on the holdout 12 months for both model and baseline.
    """
    train, test = train_test_split_series(series, test_months=12)

    features = make_features(series)
    # Warmup period is 12 months for lag_12 and roll_mean_12
    X_train = features.iloc[12:-12]
    y_train = series.iloc[12:-12]
    X_test = features.iloc[-12:]
    y_test = test.values

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    baseline_preds = seasonal_naive_predict(train, periods=12)

    # Avoid zero-division in MAPE
    denom = np.maximum(y_test, 1)
    model_mape = float(np.mean(np.abs((y_test - preds) / denom))) * 100.0
    model_rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

    baseline_mape = float(np.mean(np.abs((y_test - baseline_preds) / denom))) * 100.0
    baseline_rmse = float(np.sqrt(np.mean((y_test - baseline_preds) ** 2)))

    return {
        "model_mape": model_mape,
        "model_rmse": model_rmse,
        "baseline_mape": baseline_mape,
        "baseline_rmse": baseline_rmse,
    }


def forecast_recursive(
    series: pd.Series, periods: int = 6
) -> list[tuple[str, float]]:
    """Generate recursive multi-step ahead forecast for periods steps ahead."""
    features = make_features(series)
    X = features.iloc[12:]
    y = series.iloc[12:]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    curr_series = series.copy()
    results: list[tuple[str, float]] = []

    try:
        future_dates = pd.date_range(
            start=series.index[-1] + pd.offsets.Day(1), periods=periods, freq="ME"
        )
    except ValueError:
        future_dates = pd.date_range(
            start=series.index[-1] + pd.offsets.Day(1), periods=periods, freq="M"
        )

    for dt in future_dates:
        n = len(curr_series)
        row = {
            "lag_1": curr_series.iloc[-1],
            "lag_2": curr_series.iloc[-2],
            "lag_3": curr_series.iloc[-3],
            "lag_12": curr_series.iloc[-12],
            "roll_mean_3": curr_series.iloc[-3:].mean(),
            "roll_mean_12": curr_series.iloc[-12:].mean(),
            "month": dt.month,
            "trend": float(n),
        }
        X_step = pd.DataFrame([row])
        pred_val = float(model.predict(X_step)[0])
        results.append((dt.strftime("%Y-%m"), round(pred_val, 1)))
        curr_series = pd.concat([curr_series, pd.Series([pred_val], index=[dt])])

    return results
