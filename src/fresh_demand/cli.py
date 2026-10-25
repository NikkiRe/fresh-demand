"""Command-line entry points for data, evaluation, training and inference."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="fresh-demand")
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="Download pinned public Parquet files")
    get.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    check = commands.add_parser("audit", help="Audit one daily Parquet file")
    check.add_argument("path", type=Path)
    check.add_argument("--output", type=Path)
    subset = commands.add_parser("subset", help="Filter history to a saved train-only cohort")
    subset.add_argument("--input", type=Path, required=True)
    subset.add_argument("--series", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Backtest, freeze selection, optionally run held-out test")
    train.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    train.add_argument("--config", type=Path, default=Path("configs/default.json"))
    train.add_argument("--output", type=Path, default=Path("reports/main"))
    train.add_argument("--final-test", action="store_true")
    predict = commands.add_parser("predict", help="Predict next day from saved model and observed history")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--history", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    refit = commands.add_parser("retrain", help="Refit the frozen selected method without reading held-out data")
    refit.add_argument("--history", type=Path, required=True)
    refit.add_argument("--selection", type=Path, required=True)
    refit.add_argument("--config", type=Path, required=True)
    refit.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        from .data import download
        print(json.dumps(download(args.data_dir), indent=2))
    elif args.command == "audit":
        from .data import audit
        report = audit(args.path)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    elif args.command == "subset":
        import polars as pl
        from .data import KEYS, read_subset
        pairs = pl.read_parquet(args.series).select(KEYS).unique()
        result = read_subset(args.input, pairs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(args.output)
        print(f"Wrote {result.height} history rows from {pairs.height} saved series")
    elif args.command == "train":
        from .pipeline import run
        run(args.data_dir, args.config, args.output, final_test=args.final_test)
    elif args.command == "predict":
        from .pipeline import forecast
        result = forecast(args.model, args.history, args.output)
        print(f"Wrote {result.height} next-day forecasts to {args.output}")
    else:
        from .retrain import retrain
        print(json.dumps(retrain(args.history, args.selection, args.config, args.model), indent=2))
