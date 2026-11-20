import numpy as np
import pandas as pd
from gridcast.data import load_data
from gridcast.stationarity import adf_report


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


def test_adf_report_stationary_white_noise():
    rng = np.random.RandomState(42)
    dates = pd.date_range("2010-01-31", periods=200, freq="ME")
    white_noise = pd.Series(rng.randn(200), index=dates)

    report = adf_report(white_noise)
    assert report["stationary"] is True
    assert report["p_value"] < 0.05
