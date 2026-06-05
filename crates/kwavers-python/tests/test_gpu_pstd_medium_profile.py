#!/usr/bin/env python3
"""
Regression coverage for the GPU PSTD medium-upload split.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.timeout(30)
def test_gpu_pstd_session_reports_split_medium_profile():
    try:
        import pykwavers as pkw
    except Exception as exc:  # pragma: no cover - import-time environment guard
        pytest.skip(f"pykwavers import unavailable: {exc}")

    nx = ny = nz = 8
    dx = 1.0e-3
    dt = 1.0e-8
    nt = 8

    grid = pkw.Grid(nx, ny, nz, dx, dx, dx)
    ss_full = np.full((nx, ny, nz), 1500.0, dtype=np.float64)
    rho_full = np.full((nx, ny, nz), 1000.0, dtype=np.float64)
    absorb_full = np.zeros((nx, ny, nz), dtype=np.float64)
    bona_full = np.zeros((nx, ny, nz), dtype=np.float64)

    try:
        session = pkw.GpuPstdSession(
            grid,
            ss_full,
            rho_full,
            dt=dt,
            time_steps=nt,
            absorption=absorb_full,
            nonlinearity=bona_full,
            pml_size_xyz=(1, 1, 1),
            alpha_power=1.5,
        )
    except RuntimeError as exc:
        pytest.skip(f"GPU PSTD unavailable: {exc}")

    mask = np.zeros((nx, ny, nz), dtype=np.float64)
    mask[4, 4, 4] = 1.0
    ux_signals = np.zeros((1, nt), dtype=np.float64)
    ux_signals[0, :4] = 1.0e-3

    session.set_source_sensor(mask, ux_signals)
    session.disable_source_correction()

    ss_var = ss_full.copy()
    rho_var = rho_full.copy()
    ss_var[4, 4, 4] = 1540.0
    rho_var[4, 4, 4] = 1010.0

    sensor_data = np.asarray(session.run_scan_line(ss_var, rho_var))
    profile = session.last_run_profile
    profile_ns = session.last_run_profile_ns

    assert sensor_data.shape == (1, nt)
    assert np.max(np.abs(sensor_data)) > 0.0
    assert profile["medium_variable_upload_ns"] > 0
    assert profile["medium_static_upload_ns"] == 0
    assert profile["medium_upload_ns"] == profile["medium_variable_upload_ns"]
    assert profile["solver_run_ns"] > 0
    assert profile["materialize_ns"] >= 0
    assert profile_ns[0] == profile["medium_upload_ns"]
    assert profile_ns[1] == profile["medium_variable_upload_ns"]
    assert profile_ns[2] == profile["medium_static_upload_ns"]
    assert profile_ns[3] == profile["solver_run_ns"]
    assert profile_ns[4] == profile["materialize_ns"]
    assert profile_ns[5] == profile["total_ns"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
