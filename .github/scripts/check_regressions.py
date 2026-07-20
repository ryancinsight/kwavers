#!/usr/bin/env python3
"""Save Criterion benchmark baselines and detect regressions for Kwavers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class BenchmarkMetric:
    name: str
    mean_ns: float
    median_ns: float
    source: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "mean_ns": self.mean_ns,
            "median_ns": self.median_ns,
            "source": self.source,
        }

    @staticmethod
    def from_dict(name: str, data: Dict[str, object]) -> "BenchmarkMetric":
        return BenchmarkMetric(
            name=name,
            mean_ns=float(data["mean_ns"]),  # type: ignore[arg-type]
            median_ns=float(data["median_ns"]),  # type: ignore[arg-type]
            source=str(data["source"]),
        )


def _read_criterion_estimates(estimates_path: Path) -> Optional[Tuple[float, float]]:
    try:
        with estimates_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    mean = data.get("mean", {}).get("point_estimate")
    median = data.get("median", {}).get("point_estimate")
    if mean is None or median is None:
        return None
    return float(mean), float(median)


def _criterion_metrics(workspace: Path) -> Iterable[BenchmarkMetric]:
    criterion_root = workspace / "target" / "criterion"
    if not criterion_root.exists():
        return

    for group_dir in criterion_root.iterdir():
        if not group_dir.is_dir():
            continue
        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir():
                continue
            estimates = bench_dir / "new" / "estimates.json"
            if not estimates.exists():
                continue
            result = _read_criterion_estimates(estimates)
            if result is None:
                continue
            mean_ns, median_ns = result
            yield BenchmarkMetric(
                name=f"{group_dir.name}/{bench_dir.name}",
                mean_ns=mean_ns,
                median_ns=median_ns,
                source="criterion",
            )


def _apollo_csv_metrics(csv_path: Path) -> Iterable[BenchmarkMetric]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            case = row.get("case")
            median_ns = row.get("median_ns")
            mean_ns = row.get("min_ns")
            if not case or median_ns is None:
                continue
            try:
                yield BenchmarkMetric(
                    name=case,
                    mean_ns=float(mean_ns) if mean_ns else float(median_ns),
                    median_ns=float(median_ns),
                    source="apollo-csv",
                )
            except ValueError:
                continue


def save_baseline(workspace: Path, output: Path, csv_path: Optional[Path] = None) -> None:
    metrics: Dict[str, BenchmarkMetric] = {}

    for metric in _criterion_metrics(workspace):
        metrics[metric.name] = metric

    if csv_path is not None and csv_path.exists():
        for metric in _apollo_csv_metrics(csv_path):
            metrics[metric.name] = metric

    if not metrics:
        print(f"No benchmark metrics found in {workspace}", file=sys.stderr)
        sys.exit(1)

    baseline = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(workspace.resolve()),
        "benchmarks": {name: metric.as_dict() for name, metric in metrics.items()},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)
        fh.write("\n")

    print(f"Saved baseline with {len(metrics)} benchmarks to {output}")


def check_regression(
    workspace: Path,
    baseline_path: Path,
    threshold_pct: float,
    csv_path: Optional[Path] = None,
) -> None:
    if not baseline_path.exists():
        print(f"Baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)

    with baseline_path.open("r", encoding="utf-8") as fh:
        baseline_data = json.load(fh)

    baseline: Dict[str, BenchmarkMetric] = {
        name: BenchmarkMetric.from_dict(name, data)
        for name, data in baseline_data.get("benchmarks", {}).items()
    }

    current: Dict[str, BenchmarkMetric] = {}
    for metric in _criterion_metrics(workspace):
        current[metric.name] = metric

    if csv_path is not None and csv_path.exists():
        for metric in _apollo_csv_metrics(csv_path):
            current[metric.name] = metric

    regressions: list[Tuple[str, float, float, float]] = []
    missing: list[str] = []

    for name, base_metric in baseline.items():
        current_metric = current.get(name)
        if current_metric is None:
            missing.append(name)
            continue

        base_value = base_metric.median_ns or base_metric.mean_ns
        current_value = current_metric.median_ns or current_metric.mean_ns
        if base_value <= 0 or current_value <= 0 or math.isnan(base_value) or math.isnan(current_value):
            continue

        ratio = current_value / base_value
        if ratio > 1 + (threshold_pct / 100.0):
            regressions.append((name, base_value, current_value, ratio))

    if missing:
        print(f"Warning: {len(missing)} baseline benchmarks were not found in the current run:")
        for name in missing:
            print(f"  - {name}")

    if regressions:
        print(
            f"Regression detected ({len(regressions)} benchmark(s) exceeded "
            f"{threshold_pct}% threshold):"
        )
        for name, base, current, ratio in regressions:
            print(
                f"  - {name}: {base:.3f} ns -> {current:.3f} ns "
                f"({(ratio - 1) * 100:.1f}% regression)"
            )
        sys.exit(1)

    print(
        f"No regressions detected. Compared {len(baseline)} baseline benchmark(s) "
        f"against current run."
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Save Criterion benchmark baselines and detect regressions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save", help="Save current benchmark results as a baseline.")
    save_parser.add_argument("--workspace", type=Path, required=True, help="Path to the Rust workspace.")
    save_parser.add_argument("--output", type=Path, required=True, help="Path to write the baseline JSON.")
    save_parser.add_argument("--csv", type=Path, dest="csv_path", help="Apollo CSV report to include.")

    check_parser = subparsers.add_parser("check", help="Check current results against a baseline.")
    check_parser.add_argument("--workspace", type=Path, required=True, help="Path to the Rust workspace.")
    check_parser.add_argument("--baseline", type=Path, required=True, help="Path to the baseline JSON.")
    check_parser.add_argument(
        "--threshold-pct",
        type=float,
        default=15.0,
        help="Percentage regression threshold (default: 15).",
    )
    check_parser.add_argument("--csv", type=Path, dest="csv_path", help="Apollo CSV report to include.")

    args = parser.parse_args(argv)

    if args.command == "save":
        save_baseline(args.workspace, args.output, csv_path=args.csv_path)
    elif args.command == "check":
        check_regression(args.workspace, args.baseline, args.threshold_pct, csv_path=args.csv_path)


if __name__ == "__main__":
    main()
