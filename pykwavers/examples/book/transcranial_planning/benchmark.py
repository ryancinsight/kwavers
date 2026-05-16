from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .scene import CANONICAL_BRAIN_SCENE, BrainSceneDefinition


def run_skull_adaptive_benchmark(
    ct_path: Path,
    *,
    scene: BrainSceneDefinition = CANONICAL_BRAIN_SCENE,
    grid_size: int = 32,
    element_count: int | None = None,
    aperture_diameter_m: float | None = None,
) -> dict[str, Any]:
    import pykwavers as kw

    kwargs = scene.benchmark_pykwavers_kwargs()
    kwargs["grid_size"] = grid_size
    if element_count is not None:
        kwargs["element_count"] = element_count
    if aperture_diameter_m is not None:
        kwargs["aperture_diameter_m"] = aperture_diameter_m
    result = kw.run_transcranial_skull_adaptive_benchmark_from_ritk_ct(
        str(ct_path),
        **kwargs,
    )
    return summarize_benchmark_result(result)


def summarize_benchmark_result(result: dict[str, Any]) -> dict[str, Any]:
    active = np.asarray(result["active_elements"], dtype=bool)
    metrics = result["metrics"]
    placement = result["placement"]
    return {
        "benchmark_model": result["benchmark_model"],
        "frequency_hz": float(result["frequency_hz"]),
        "target_peak_pa": float(result["target_peak_pa"]),
        "focus_index": [int(v) for v in result["focus_index"]],
        "target_fraction_xyz": [float(v) for v in result.get("target_fraction_xyz", ())],
        "active_element_count": int(np.count_nonzero(active)),
        "aperture_diameter_m": float(placement["aperture_diameter_m"]),
        "radius_of_curvature_m": float(placement["radius_of_curvature_m"]),
        "mean_skull_length_m": float(placement["mean_skull_length_m"]),
        "mean_amplitude_weight": float(placement["mean_amplitude_weight"]),
        "relative_l2": float(metrics["relative_l2"]),
        "focal_position_error_m": float(metrics["focal_position_error_m"]),
        "max_pressure_error_percent": float(metrics["max_pressure_error_percent"]),
        "paper_structural_comparison": dict(result["paper_structural_comparison"]),
    }


def write_skull_adaptive_benchmark_summary(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summarize_benchmark_result(result), indent=2), encoding="utf-8")
