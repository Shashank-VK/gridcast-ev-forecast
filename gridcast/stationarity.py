import warnings
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller


def adf_report(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller test on a series.

    Returns dict with keys:
    - statistic: ADF test statistic (float)
    - p_value: approximate p-value (float)
    - stationary: bool (True if p_value < 0.05, else False)
    """
    clean_series = series.dropna()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        res = adfuller(clean_series)
    stat = float(res[0])
    p_val = float(res[1])
    return {
        "statistic": stat,
        "p_value": p_val,
        "stationary": bool(p_val < 0.05),
    }


def acf_report(series: pd.Series) -> dict:
    """Autocorrelation of a series at the lags that matter for monthly data.

    Returns dict with keys:
    - lag_1: autocorrelation at lag 1 month (float in [-1, 1])
    - lag_12: autocorrelation at lag 12 months, the seasonal lag (float)
    """
    clean_series = series.dropna()
    values = acf(clean_series, nlags=12, fft=True)
    return {
        "lag_1": float(values[1]),
        "lag_12": float(values[12]),
    }
