import argparse
import sys
from gridcast.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gridcast",
        description="GridCast — EV Fleet Growth Forecaster",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")
    run_parser = subparsers.add_parser("run", help="Run model evaluation and forecasting pipeline")
    run_parser.add_argument(
        "--metrics-output",
        default="outputs/metrics.json",
        help="Path for metrics.json output",
    )
    run_parser.add_argument(
        "--forecast-output",
        default="outputs/forecast.csv",
        help="Path for forecast.csv output",
    )

    args = parser.parse_args()

    # Default to running pipeline if command is "run" or omitted
    if args.command in (None, "run"):
        metrics_out = getattr(args, "metrics_output", "outputs/metrics.json")
        forecast_out = getattr(args, "forecast_output", "outputs/forecast.csv")
        run_pipeline(metrics_output=metrics_out, forecast_output=forecast_out)
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
