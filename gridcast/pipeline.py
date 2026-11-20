import json
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd

from gridcast.data import load_data
from gridcast.model import evaluate_zone, forecast_recursive
from gridcast.stationarity import adf_report


def run_pipeline(
    data_path: Union[str, Path, None] = None,
    metrics_output: Union[str, Path] = "outputs/metrics.json",
    forecast_output: Union[str, Path] = "outputs/forecast.csv",
) -> dict:
    """Run full GridCast evaluation and forecasting pipeline.

    1. Loads EV population series per zone.
    2. Filters zones with at least 48 months of history.
    3. Evaluates GradientBoostingRegressor vs seasonal-naive baseline.
    4. Computes ADF stationarity report per zone.
    5. Writes outputs/metrics.json.
    6. Generates 6-month recursive forecast for top 5 zones and writes outputs/forecast.csv.
    """
    metrics_path = Path(metrics_output)
    forecast_path = Path(forecast_output)

    # 1. Load data
    zones = load_data(data_path)

    # 2. Filter zones with at least 48 months
    evaluated_zones = {z: s for z, s in zones.items() if len(s) >= 48}

    if not evaluated_zones:
        raise ValueError("No zones with at least 48 months of history found.")

    # Compute overall data range across evaluated zones
    min_date = min(s.index.min() for s in evaluated_zones.values())
    max_date = max(s.index.max() for s in evaluated_zones.values())
    data_range = {
        "start": min_date.strftime("%Y-%m-%d"),
        "end": max_date.strftime("%Y-%m-%d"),
    }

    per_zone: dict[str, dict[str, float]] = {}
    adf_dict: dict[str, dict] = {}

    for zone, series in sorted(evaluated_zones.items()):
        # Holdout evaluation
        metrics = evaluate_zone(series)
        per_zone[zone] = {
            "model_mape": round(metrics["model_mape"], 4),
            "model_rmse": round(metrics["model_rmse"], 4),
            "baseline_mape": round(metrics["baseline_mape"], 4),
            "baseline_rmse": round(metrics["baseline_rmse"], 4),
        }

        # ADF stationarity test
        adf_res = adf_report(series)
        adf_dict[zone] = {
            "statistic": round(adf_res["statistic"], 4),
            "p_value": round(adf_res["p_value"], 6),
            "stationary": adf_res["stationary"],
        }

    # Overall metrics across all evaluated zones
    overall = {
        "model_mape": round(
            float(np.mean([m["model_mape"] for m in per_zone.values()])), 4
        ),
        "model_rmse": round(
            float(np.mean([m["model_rmse"] for m in per_zone.values()])), 4
        ),
        "baseline_mape": round(
            float(np.mean([m["baseline_mape"] for m in per_zone.values()])), 4
        ),
        "baseline_rmse": round(
            float(np.mean([m["baseline_rmse"] for m in per_zone.values()])), 4
        ),
    }

    metrics_payload = {
        "data_range": data_range,
        "zones_evaluated": len(evaluated_zones),
        "overall": overall,
        "per_zone": per_zone,
        "adf": adf_dict,
    }

    # Write metrics.json
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 6-month recursive forecast for top 5 zones by latest EV population
    top_5_zones = sorted(
        evaluated_zones.keys(),
        key=lambda z: evaluated_zones[z].iloc[-1],
        reverse=True,
    )[:5]

    forecast_rows = []
    for zone in top_5_zones:
        series = evaluated_zones[zone]
        predictions = forecast_recursive(series, periods=6)
        for month_str, pred_val in predictions:
            forecast_rows.append(
                {
                    "zone": zone,
                    "month": month_str,
                    "predicted_ev_population": pred_val,
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_path, index=False)

    print(f"[gridcast] Evaluated {len(evaluated_zones)} zones.")
    print(
        f"[gridcast] Overall Model MAPE: {overall['model_mape']:.2f}% vs Baseline MAPE: {overall['baseline_mape']:.2f}%"
    )
    print(
        f"[gridcast] Overall Model RMSE: {overall['model_rmse']:.2f} vs Baseline RMSE: {overall['baseline_rmse']:.2f}"
    )
    print(f"[gridcast] Wrote metrics to {metrics_path}")
    print(f"[gridcast] Wrote 6-month forecast to {forecast_path}")

    return metrics_payload
