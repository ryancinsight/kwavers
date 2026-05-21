import importlib
import sys
from pathlib import Path

import numpy as np


def _load_discrete_green():
    root = Path(__file__).resolve().parents[1] / "examples"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("ali2025_breast_fwi.discrete_green")


def test_pstd_periodic_pressure_trace_matches_zero_mode_cumulative_source():
    module = _load_discrete_green()
    spacing = 0.01
    sound_speed = 2.0
    time_step = 0.001
    frequency = 5.0
    source_amplitude = 3.0
    steps = 8

    traces = module.pstd_periodic_pressure_traces(
        (1, 1, 1),
        spacing,
        sound_speed,
        time_step,
        frequency,
        steps,
        [(0, 0, 0)],
        [(0, 0, 0)],
        source_amplitude,
    )

    signal = np.sin(2.0 * np.pi * frequency * time_step * np.arange(steps))
    gain = 2.0 * sound_speed * time_step * source_amplitude / spacing
    np.testing.assert_allclose(traces[0], gain * np.cumsum(signal), atol=1.0e-14)


def test_pstd_periodic_observation_cube_samples_snapped_receivers():
    module = _load_discrete_green()
    elements = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    frequency = 1.0
    time_step = 0.1
    steps = 6
    bin_start = 2
    source_amplitude = 2.0

    cube = module.pstd_periodic_observation_cube(
        elements,
        2,
        1,
        [frequency],
        1.0,
        1.0,
        (2, 1, 1),
        time_step,
        [steps],
        [bin_start],
        source_amplitude,
    )
    traces = module.pstd_periodic_pressure_traces(
        (2, 1, 1),
        1.0,
        1.0,
        time_step,
        frequency,
        steps,
        [(0, 0, 0)],
        [(0, 0, 0), (1, 0, 0)],
        source_amplitude,
    )
    expected = module.frequency_bin_complex_traces(traces, frequency, time_step, bin_start)

    np.testing.assert_allclose(cube[0, 0, :], expected, atol=1.0e-14)
