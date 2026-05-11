from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "histotripsy_cavitation_compare.py"
EXAMPLES_DIR = EXAMPLE_PATH.parent


def load_example_module():
    sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location("histotripsy_cavitation_compare", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ms_pressure_response_has_physical_zero_and_monotone_collapse_strength():
    module = load_example_module()
    config = module.HistotripsyConfig(bubble_pressure_samples=16, bubble_response_cycles=4)

    pressure, radius_ratio, gas_temperature = module.ms_pressure_response_curve(config)

    assert pressure[0] == 0.0
    assert radius_ratio[0] == 1.0
    assert gas_temperature[0] == config.bubble_temperature_baseline_k
    assert np.all(np.isfinite(radius_ratio))
    assert np.all(np.isfinite(gas_temperature))
    assert np.all(np.diff(radius_ratio) >= -1.0e-12)
    assert gas_temperature[-1] > gas_temperature[0]


def test_focal_support_metrics_reject_axisymmetric_cylinder_regression():
    module = load_example_module()
    grid = module.build_volume_grid()
    intensity = module.axisymmetric_intensity_volume(grid)
    metrics = module.focal_support_metrics(intensity, grid, level=0.5)

    assert metrics["lateral_diameter_mm"] > 0.0
    assert metrics["axial_length_mm"] > metrics["lateral_diameter_mm"]
    assert metrics["axial_to_lateral_ratio"] < 10.0
    assert 33.0 <= metrics["focus_z_mm"] <= 36.0


def test_ms_cavitation_maps_are_finite_normalized_and_focal():
    module = load_example_module()
    config = module.HistotripsyConfig(bubble_pressure_samples=16, bubble_response_cycles=4)
    grid = module.build_volume_grid()
    intensity = module.axisymmetric_intensity_volume(grid)

    radius_ratio, activity, gas_temperature = module.ms_pulse_cavitation_maps(intensity, config)

    assert np.all(np.isfinite(radius_ratio))
    assert np.all(np.isfinite(activity))
    assert np.all(np.isfinite(gas_temperature))
    assert np.min(activity) >= 0.0
    assert np.max(activity) == 1.0
    assert np.unravel_index(int(np.argmax(activity)), activity.shape) == np.unravel_index(
        int(np.argmax(intensity)), intensity.shape
    )
