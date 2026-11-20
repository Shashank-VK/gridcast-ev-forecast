import numpy as np
import pandas as pd
from gridcast.data import load_data
from gridcast.stationarity import acf_report, adf_report


def test_adf_report_king_county():
    zones_data = load_data()
    king_series = zones_data["King"]
    report = adf_report(king_series)

    assert isinstance(report, dict)
    assert "statistic" in report
    assert "p_value" in report
    assert "stationary" in report
    assert isinstance(report["statistic"], float)
    assert isinstance(report["p_value"], float)
    assert isinstance(report["stationary"], bool)
    assert report["stationary"] == (report["p_value"] < 0.05)


def test_acf_report_ar1_and_king():
    rng = np.random.RandomState(7)
    dates = pd.date_range("2010-01-31", periods=240, freq="ME")
    noise = rng.randn(240)
    ar1 = pd.Series(np.convolve(noise, [1.0], mode="same"), index=dates)
    # Build a genuine AR(1) with phi = 0.9
    vals = np.zeros(240)
    for t in range(1, 240):
        vals[t] = 0.9 * vals[t - 1] + noise[t]
    ar1 = pd.Series(vals, index=dates)

    report = acf_report(ar1)
    assert set(report.keys()) == {"lag_1", "lag_12"}
    assert -1.0 <= report["lag_1"] <= 1.0
    assert report["lag_1"] > 0.7

    king_report = acf_report(load_data()["King"])
    assert -1.0 <= king_report["lag_12"] <= 1.0
    assert king_report["lag_12"] > 0.5


def test_adf_report_stationary_white_noise():
    rng = np.random.RandomState(42)
    dates = pd.date_range("2010-01-31", periods=200, freq="ME")
    white_noise = pd.Series(rng.randn(200), index=dates)

    report = adf_report(white_noise)
    assert report["stationary"] is True
    assert report["p_value"] < 0.05
