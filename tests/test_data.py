import pytest
import pandas as pd
from gridcast.data import load_data, train_test_split_series


def test_load_data_row_count():
    zones_data = load_data()
    assert isinstance(zones_data, dict)
    assert len(zones_data) >= 30
    total_rows = sum(len(s) for s in zones_data.values())
    assert total_rows > 4000


def test_data_monthly_sorted_index():
    zones_data = load_data()
    for zone, series in zones_data.items():
        assert isinstance(series, pd.Series)
        assert isinstance(series.index, pd.DatetimeIndex)
        assert series.index.is_monotonic_increasing
        assert series.index.is_unique
        # Values must be integer counts
        assert pd.api.types.is_integer_dtype(series.dtype)
        # Check no gaps: differences between adjacent month-ends must be 28 to 31 days
        day_diffs = series.index.to_series().diff().dropna().dt.days
        assert (day_diffs >= 28).all() and (day_diffs <= 31).all()


def test_train_test_split_leakage_guard():
    zones_data = load_data()
    king_series = zones_data["King"]
    test_months = 12
    train, test = train_test_split_series(king_series, test_months=test_months)

    assert len(test) == test_months
    assert len(train) == len(king_series) - test_months
    # Train strictly ends before test begins
    assert train.index.max() < test.index.min()
    # No overlapping dates
    assert set(train.index).isdisjoint(set(test.index))
    # Full concatenation matches original
    assert len(train) + len(test) == len(king_series)


def test_train_test_split_invalid_size():
    zones_data = load_data()
    series = zones_data["King"].iloc[:10]
    with pytest.raises(ValueError):
        train_test_split_series(series, test_months=12)
