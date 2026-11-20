from pathlib import Path
from typing import Union
import pandas as pd


def load_data(
    path: Union[str, Path, None] = None
) -> dict[str, pd.Series]:
    """Load EV population history and return dict of zone -> monthly pandas Series.

    Each series has a sorted, month-end DatetimeIndex without gaps,
    reindexed and forward-filled within the zone's observed span.
    """
    if path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "ev_population_by_county_raw.csv"
    else:
        csv_path = Path(path)

    df = pd.read_csv(csv_path)

    # Parse dates and normalize to month-end timestamps
    dates = pd.to_datetime(df["date"])
    df["date"] = dates.dt.to_period("M").dt.to_timestamp("M")

    # Coerce counts to integer
    df["electric_vehicle_ev_total"] = pd.to_numeric(
        df["electric_vehicle_ev_total"], errors="coerce"
    ).fillna(0).astype(int)

    zones_dict: dict[str, pd.Series] = {}

    for zone, group in df.groupby("county"):
        zone_str = str(zone)
        grp = group.sort_values("date")
        s = grp.set_index("date")["electric_vehicle_ev_total"]

        min_date = s.index.min()
        max_date = s.index.max()

        try:
            full_range = pd.date_range(start=min_date, end=max_date, freq="ME")
        except ValueError:
            full_range = pd.date_range(start=min_date, end=max_date, freq="M")

        # Reindex and forward-fill only within observed span
        s_reindexed = s.reindex(full_range).ffill().bfill().astype(int)
        s_reindexed.index.name = "date"
        s_reindexed.name = zone_str

        zones_dict[zone_str] = s_reindexed

    return zones_dict


# Convenience aliases
load_ev_data = load_data
load_zone_series = load_data


def train_test_split_series(
    series: pd.Series, test_months: int = 12
) -> tuple[pd.Series, pd.Series]:
    """Chronologically split a series into (train, test) with no leakage."""
    if len(series) <= test_months:
        raise ValueError(
            f"Series length ({len(series)}) must be greater than test_months ({test_months})"
        )
    train = series.iloc[:-test_months]
    test = series.iloc[-test_months:]
    return train, test
