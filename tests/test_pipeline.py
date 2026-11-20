import json
from pathlib import Path
import pandas as pd
from gridcast.pipeline import run_pipeline


def test_pipeline_outputs(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    forecast_path = tmp_path / "forecast.csv"

    run_pipeline(metrics_output=metrics_path, forecast_output=forecast_path)

    assert metrics_path.exists()
    assert forecast_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Schema checks
    assert "data_range" in metrics
    assert "start" in metrics["data_range"]
    assert "end" in metrics["data_range"]
    assert "zones_evaluated" in metrics
    assert metrics["zones_evaluated"] >= 20

    assert "overall" in metrics
    for k in ["model_mape", "model_rmse", "baseline_mape", "baseline_rmse"]:
        assert k in metrics["overall"]
        assert isinstance(metrics["overall"][k], (int, float))
        assert metrics["overall"][k] >= 0

    assert "per_zone" in metrics
    assert len(metrics["per_zone"]) == metrics["zones_evaluated"]

    assert "adf" in metrics
    assert len(metrics["adf"]) == metrics["zones_evaluated"]
    for zone, adf_res in metrics["adf"].items():
        assert "statistic" in adf_res
        assert "p_value" in adf_res
        assert "stationary" in adf_res

    # Forecast checks
    forecast_df = pd.read_csv(forecast_path)
    expected_cols = ["zone", "month", "predicted_ev_population"]
    assert list(forecast_df.columns) == expected_cols
    # Exactly 5 zones * 6 months = 30 rows
    assert len(forecast_df) == 30
    assert forecast_df["zone"].nunique() == 5
    assert (forecast_df["predicted_ev_population"] > 0).all()
