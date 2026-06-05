"""Patient-adaptive transmit scheduling experiment for Chapter 29."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ADAPTIVE_TRANSMIT_FIGURE_NAME = "fig07_patient_adaptive_transmit_budget.png"
ADAPTIVE_TRANSMIT_METRICS_NAME = "adaptive_transmit_metrics.json"
SCHEDULE_STRATEGIES = ("uniform", "patient_adaptive")
DEFAULT_BUDGET_FRACTIONS = (0.125, 0.25)


def run_adaptive_transmit_comparison(
    cases: Iterable[dict[str, object]],
    out_dir: Path,
    runner: Callable[..., dict[str, object]],
    scene_kwargs_for_case: Callable[[dict[str, object]], dict[str, object]],
    save_figure: Callable[[plt.Figure, Path], None],
) -> tuple[Path, Path, list[dict[str, object]]]:
    records = collect_adaptive_transmit_records(cases, runner, scene_kwargs_for_case)
    figure = render_adaptive_transmit_records(records, out_dir, save_figure)
    metrics = write_adaptive_transmit_metrics(records, out_dir, figure)
    return figure, metrics, records


def collect_adaptive_transmit_records(
    cases: Iterable[dict[str, object]],
    runner: Callable[..., dict[str, object]],
    scene_kwargs_for_case: Callable[[dict[str, object]], dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for case in cases:
        for budget in transmit_budget_values(case):
            for strategy in SCHEDULE_STRATEGIES:
                result = run_scheduled_case(case, runner, scene_kwargs_for_case, strategy, budget)
                records.append(record_from_result(result, strategy, budget))
        if include_full_reference():
            result = run_scheduled_case(case, runner, scene_kwargs_for_case, "full", None)
            records.append(record_from_result(result, "full", None))
    return records


def run_scheduled_case(
    case: dict[str, object],
    runner: Callable[..., dict[str, object]],
    scene_kwargs_for_case: Callable[[dict[str, object]], dict[str, object]],
    strategy: str,
    budget: int | None,
) -> dict[str, object]:
    seg = case["seg"]
    return runner(
        str(case["ct"]),
        None if seg is None else str(seg),
        anatomy=str(case["name"]),
        grid_size=int(os.environ.get("KWAVERS_CH29_ADAPTIVE_GRID", str(case["grid"]))),
        element_count=int(case["elements"]),
        iterations=int(os.environ.get("KWAVERS_CH29_ADAPTIVE_ITERATIONS", "6")),
        frequencies_hz=list(case["freq"]),
        receiver_offsets=list(case["offsets"]),
        source_pressure_pa=float(case["pressure"]),
        noise_fraction=float(os.environ.get("KWAVERS_CH29_NOISE_FRACTION", "0.012")),
        inverse_encoding_rows_per_code=int(os.environ.get("KWAVERS_CH29_INVERSE_ENCODING_ROWS_PER_CODE", "2")),
        transmit_schedule_strategy=strategy,
        transmit_budget=budget,
        **scene_kwargs_for_case(case),
    )


def record_from_result(result: dict[str, object], strategy: str, requested_budget: int | None) -> dict[str, object]:
    active = metric_block(result, "active_lesion")
    fusion = metric_block(result, "fusion")
    return {
        "anatomy": str(result["anatomy"]),
        "strategy": str(result.get("transmit_schedule_strategy", strategy)),
        "requested_budget": None if requested_budget is None else int(requested_budget),
        "effective_budget": int(result.get("transmit_budget_effective", requested_budget or result["element_count"])),
        "budget_fraction": float(result.get("transmit_budget_fraction", 1.0)),
        "encoded_measurements": int(result["encoded_measurements"]),
        "unencoded_measurements": int(result["unencoded_measurements"]),
        "active_dice_equal_area": float(active["dice_equal_area"]),
        "active_cnr": float(active["cnr"]),
        "fusion_dice_equal_area": float(fusion["dice_equal_area"]),
        "fusion_cnr": float(fusion["cnr"]),
        "sequence_indices": [int(v) for v in result.get("transmit_sequence_indices", [])],
    }


def render_adaptive_transmit_records(
    records: list[dict[str, object]],
    out_dir: Path,
    save_figure: Callable[[plt.Figure, Path], None],
) -> Path:
    anatomies = sorted({str(record["anatomy"]) for record in records})
    fig, axes = plt.subplots(1, len(anatomies), figsize=(5.2 * len(anatomies), 4.2), constrained_layout=True)
    axes_1d = np.asarray(axes, dtype=object).reshape(-1)
    styles = {
        "uniform": {"marker": "o", "color": "#3b6fb6", "label": "uniform"},
        "patient_adaptive": {"marker": "s", "color": "#c44e52", "label": "patient-adaptive"},
        "full": {"marker": "^", "color": "#2f7d32", "label": "full reference"},
    }
    for ax, anatomy in zip(axes_1d, anatomies):
        subset = [record for record in records if record["anatomy"] == anatomy]
        for strategy, style in styles.items():
            series = sorted(
                (record for record in subset if record["strategy"] == strategy),
                key=lambda record: int(record["effective_budget"]),
            )
            if not series:
                continue
            x = [int(record["effective_budget"]) for record in series]
            y = [float(record["active_dice_equal_area"]) for record in series]
            ax.plot(x, y, linewidth=1.2, markersize=4.0, **style)
        ax.set_title(anatomy)
        ax.set_xlabel("focused transmit budget")
        ax.set_ylabel("active inverse Dice")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    path = out_dir / ADAPTIVE_TRANSMIT_FIGURE_NAME
    save_figure(fig, path)
    plt.close(fig)
    return path


def write_adaptive_transmit_metrics(
    records: list[dict[str, object]],
    out_dir: Path,
    figure: Path | None,
) -> Path:
    payload = adaptive_transmit_payload(records, figure)
    path = out_dir / ADAPTIVE_TRANSMIT_METRICS_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def adaptive_transmit_payload(records: list[dict[str, object]], figure: Path | None) -> dict[str, object]:
    return {
        "experiment": "patient_adaptive_focused_transmit_scheduling",
        "paper_inspiration": {
            "arxiv": "2508.08782",
            "principle": (
                "select a sparse patient-conditioned focused transmit sequence and compare "
                "reconstruction fidelity against an equispaced sequence at matched budget"
            ),
        },
        "control_surface": {
            "minimal_parameters": ["transmit_schedule_strategy", "transmit_budget"],
            "fixed_pipeline": (
                "CT preprocessing, device placement, finite-frequency operators, row encoding, "
                "PCG solve, channel fusion, and metrics remain unchanged"
            ),
        },
        "summary": adaptive_summary(records),
        "records": records,
        "figure": None if figure is None else str(figure),
    }


def adaptive_summary(records: list[dict[str, object]]) -> dict[str, object]:
    matched_deltas = []
    for anatomy in sorted({str(record["anatomy"]) for record in records}):
        budgets = sorted({
            int(record["effective_budget"])
            for record in records
            if record["anatomy"] == anatomy and record["strategy"] != "full"
        })
        for budget in budgets:
            uniform = find_record(records, anatomy, budget, "uniform")
            adaptive = find_record(records, anatomy, budget, "patient_adaptive")
            if uniform is not None and adaptive is not None:
                matched_deltas.append(
                    float(adaptive["active_dice_equal_area"]) - float(uniform["active_dice_equal_area"])
                )
    return {
        "case_count": len({str(record["anatomy"]) for record in records}),
        "record_count": len(records),
        "mean_patient_adaptive_minus_uniform_active_dice": float(np.mean(matched_deltas))
        if matched_deltas
        else 0.0,
        "budgets_per_case": {
            anatomy: sorted({
                int(record["effective_budget"])
                for record in records
                if record["anatomy"] == anatomy
            })
            for anatomy in sorted({str(record["anatomy"]) for record in records})
        },
    }


def find_record(
    records: list[dict[str, object]],
    anatomy: str,
    budget: int,
    strategy: str,
) -> dict[str, object] | None:
    for record in records:
        if (
            record["anatomy"] == anatomy
            and int(record["effective_budget"]) == budget
            and record["strategy"] == strategy
        ):
            return record
    return None


def transmit_budget_values(case: dict[str, object]) -> list[int]:
    count = int(case["elements"])
    absolute = os.environ.get("KWAVERS_CH29_ADAPTIVE_TRANSMIT_BUDGETS")
    if absolute:
        values = [int(value.strip()) for value in absolute.split(",") if value.strip()]
        return sorted({min(max(value, 1), count) for value in values})
    fractions = budget_fractions()
    return sorted({min(max(int(round(count * fraction)), 1), count) for fraction in fractions})


def budget_fractions() -> tuple[float, ...]:
    raw = os.environ.get("KWAVERS_CH29_ADAPTIVE_TRANSMIT_FRACTIONS")
    if raw is None:
        return DEFAULT_BUDGET_FRACTIONS
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    return tuple(value for value in values if value > 0.0)


def include_full_reference() -> bool:
    return os.environ.get("KWAVERS_CH29_ADAPTIVE_INCLUDE_FULL", "1") not in {"0", "false", "False"}


def metric_block(result: dict[str, object], key: str) -> dict[str, Any]:
    metrics = result["metrics"]
    if not isinstance(metrics, dict):
        metrics = dict(metrics)
    block = metrics[key]
    return dict(block) if not isinstance(block, dict) else block
