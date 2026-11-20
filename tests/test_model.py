import numpy as np
import pandas as pd
from gridcast.data import load_data
from gridcast.model import (
    make_features,
    seasonal_naive_predict,
    evaluate_zone,
)


def test_make_features_structure_and_warmup():
    zones_data = load_data()
    series = zones_data["King"]
    features = make_features(series)

    expected_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "roll_mean_3",
        "roll_mean_12",
        "month",
        "trend",
    ]
    for col in expected_cols:
        assert col in features.columns

    # Warmup period is 12 months (due to lag_12 and roll_mean_12)
    # Warmup rows have NaNs
    assert features.iloc[:12].isna().any().any()
    # All rows after 12-month warmup must have NO NaN
    post_warmup = features.iloc[12:]
    assert post_warmup.isna().sum().sum() == 0
    assert len(post_warmup) == len(series) - 12


def test_seasonal_naive_predict():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    vals = np.arange(100, 124)
    train_series = pd.Series(vals, index=dates)

    # 12 periods repeat the last 12 values exactly
    pred_12 = seasonal_naive_predict(train_series, periods=12)
    np.testing.assert_array_equal(pred_12, train_series.iloc[-12:].values)

    # 6 periods repeat the first 6 of the last 12 values
    pred_6 = seasonal_naive_predict(train_series, periods=6)
    np.testing.assert_array_equal(pred_6, train_series.iloc[-12:-6].values)


def test_evaluate_zone_metrics():
    zones_data = load_data()
    series = zones_data["King"]
    metrics = evaluate_zone(series)

    for key in ["model_mape", "model_rmse", "baseline_mape", "baseline_rmse"]:
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert np.isfinite(metrics[key])
        assert metrics[key] >= 0

    # GradientBoosting model should outperform seasonal naive baseline
    assert metrics["model_mape"] < metrics["baseline_mape"]
    assert metrics["model_rmse"] < metrics["baseline_rmse"]
