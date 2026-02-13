#!/usr/bin/env python3
"""
Focused diagnostics for pykwavers vs k-wave-python differences.

This module is intentionally diagnostic-first (not strict pass/fail parity).
It quantifies where mismatch is coming from:
- waveform shape mismatch (raw correlation)
- timing/phase lag mismatch (best-lag correlation)
- amplitude scaling mismatch (RMS ratio)
"""

import os

import numpy as np
import pytest

from pykwavers.comparison import SimulationConfig, run_kwave_python, run_pykwavers
from pykwavers.kwave_python_bridge import KWAVE_PYTHON_AVAILABLE, compute_error_metrics

if not KWAVE_PYTHON_AVAILABLE:
    pytest.skip("k-wave-python not available", allow_module_level=True)

if os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1":
    pytest.skip("KWAVERS_SKIP_KWAVE=1 set; skipping k-wave diagnostics", allow_module_level=True)

if os.getenv("KWAVERS_RUN_SLOW", "0") != "1":
    pytest.skip("Set KWAVERS_RUN_SLOW=1 to run slow k-wave diagnostics", allow_module_level=True)


def _build_config(
    grid_shape=(32, 32, 32),
    spacing=0.2e-3,
    sound_speed=1500.0,
    density=1000.0,
    frequency=1e6,
    amplitude=1e5,
    duration=6e-6,
    source_position=None,
    sensor_position=(3.2e-3, 3.2e-3, 4.0e-3),
    pml_size=6,
):
    dx = spacing
    cfl = 0.3
    dt = cfl * dx / sound_speed

    return SimulationConfig(
        grid_shape=grid_shape,
        grid_spacing=(dx, dx, dx),
        sound_speed=sound_speed,
        density=density,
        source_frequency=frequency,
        source_amplitude=amplitude,
        duration=duration,
        source_position=source_position,
        sensor_position=sensor_position,
        pml_size=pml_size,
        dt=dt,
    )


def _align_by_lag(reference: np.ndarray, test: np.ndarray, lag: int):
    if lag == 0:
        return reference, test

    if lag > 0:
        # test is delayed; drop prefix from test and suffix from reference
        return reference[:-lag], test[lag:]

    lag = -lag
    # test is advanced; drop prefix from reference and suffix from test
    return reference[lag:], test[:-lag]


def _estimate_best_lag(reference: np.ndarray, test: np.ndarray, max_lag: int = 256):
    ref = reference.flatten().astype(np.float64)
    tst = test.flatten().astype(np.float64)

    n = min(ref.size, tst.size)
    ref = ref[:n]
    tst = tst[:n]

    ref = ref - np.mean(ref)
    tst = tst - np.mean(tst)

    best_lag = 0
    best_corr = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        r, t = _align_by_lag(ref, tst, lag)
        if r.size < 8:
            continue
        denom = np.linalg.norm(r) * np.linalg.norm(t)
        if denom <= 0:
            continue
        c = float(np.dot(r, t) / denom)
        if c > best_corr:
            best_corr = c
            best_lag = lag

    return best_lag, best_corr


@pytest.mark.parametrize(
    "scenario,solver_type,config",
    [
        ("plane_wave", "fdtd", _build_config()),
        ("plane_wave", "pstd", _build_config()),
        (
            "point_source",
            "fdtd",
            _build_config(
                duration=8e-6,
                source_position=(4 * 0.2e-3, 16 * 0.2e-3, 16 * 0.2e-3),
                sensor_position=(24 * 0.2e-3, 16 * 0.2e-3, 16 * 0.2e-3),
            ),
        ),
        (
            "point_source",
            "pstd",
            _build_config(
                duration=8e-6,
                source_position=(4 * 0.2e-3, 16 * 0.2e-3, 16 * 0.2e-3),
                sensor_position=(24 * 0.2e-3, 16 * 0.2e-3, 16 * 0.2e-3),
            ),
        ),
    ],
)
def test_difference_diagnostics(scenario: str, solver_type: str, config: SimulationConfig):

    result_kw = run_kwave_python(config)
    result_py = run_pykwavers(config, solver_type=solver_type)

    ref = np.asarray(result_kw.pressure).flatten().astype(np.float64)
    tst = np.asarray(result_py.pressure).flatten().astype(np.float64)

    n = min(ref.size, tst.size)
    ref = ref[:n]
    tst = tst[:n]

    assert np.all(np.isfinite(ref))
    assert np.all(np.isfinite(tst))
    assert np.max(np.abs(ref)) > 0.0
    assert np.max(np.abs(tst)) > 0.0

    raw = compute_error_metrics(ref, tst)

    max_lag = min(256, max(8, n // 4))
    best_lag, best_lag_corr = _estimate_best_lag(ref, tst, max_lag=max_lag)
    ref_aligned, tst_aligned = _align_by_lag(ref, tst, best_lag)
    aligned = compute_error_metrics(ref_aligned, tst_aligned)

    rms_ref = float(np.sqrt(np.mean(ref**2)))
    rms_tst = float(np.sqrt(np.mean(tst**2)))
    rms_ratio = rms_tst / max(rms_ref, 1e-30)

    print(f"\n[Diagnostics:{scenario}:{solver_type}] n={n}, max_lag={max_lag}")
    print(
        f"  Raw:     L2={raw['l2_error']:.3f}, Linf={raw['linf_error']:.3f}, corr={raw['correlation']:.3f}"
    )
    print(
        f"  Aligned: L2={aligned['l2_error']:.3f}, Linf={aligned['linf_error']:.3f}, corr={aligned['correlation']:.3f}"
    )
    print(f"  Best lag samples: {best_lag} (corr={best_lag_corr:.3f})")
    print(f"  RMS ratio (py/kw): {rms_ratio:.3f}")

    # Diagnostic sanity checks only
    assert -max_lag <= best_lag <= max_lag
    assert np.isfinite(rms_ratio)
