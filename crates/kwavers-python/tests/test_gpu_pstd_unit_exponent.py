#!/usr/bin/env python3
"""
A unit power-law exponent without dispersion through the GPU PSTD session.

``tan(pi*y/2)`` is singular at ``y = 1``, so the causal form rejects it; with
``PowerLawDispersion.Omitted`` the dispersive term is zero and ``y = 1`` is a
valid absorption. The session and the CPU PSTD ``Simulation`` build their
operators from the same Rust power-law construction, so they must agree on one
sensor trace, to the single-precision bound of
``crates/kwavers-gpu/tests/pstd_parity.rs``.
"""

from __future__ import annotations

import numpy as np
import pytest

N = 32
DX = 1.0e-3
C0 = 1500.0
RHO0 = 1000.0
DT = 0.3 * DX / C0
NT = 64
# The sensor sits SENSOR_OFFSET cells downstream of the velocity source, so the
# pulse crosses a measurable absorption path.
SENSOR_OFFSET = 8
PML = 4
ALPHA_COEFF_DB = 5.0
# max |cpu - gpu| over traces normalised to a unit lossless peak. The Rust
# parity test bounds f32 PSTD against the f64 CPU path at gamma_(100 x 216) <
# 1.3e-3 for 100 steps of a 64^3 grid and asserts 2e-3; this 32^3 grid runs
# shorter transforms for fewer steps, so the same assertion bound applies.
PARITY_BOUND = 2.0e-3
# The absorbing and lossless traces must differ by far more than the bound, or
# the comparison would not see the absorption operators (measured 0.12).
ABSORPTION_VISIBILITY = 2.0e-2


def _pykwavers():
    try:
        import pykwavers as pkw
    except Exception as exc:  # pragma: no cover - import-time environment guard
        pytest.skip(f"pykwavers import unavailable: {exc}")
    return pkw


def _velocity_pulse() -> np.ndarray:
    """A smooth x-velocity pulse of 16 steps, zero afterwards."""
    steps = np.arange(NT, dtype=np.float64)
    pulse = np.where(steps < 16, np.sin(np.pi * steps / 16.0) ** 2, 0.0)
    return 1.0e-3 * pulse


def _voxel_mask(ix: int) -> np.ndarray:
    mask = np.zeros((N, N, N), dtype=np.float64)
    mask[ix, N // 2, N // 2] = 1.0
    return mask


SOURCE_IX = N // 2 - SENSOR_OFFSET // 2
SENSOR_IX = SOURCE_IX + SENSOR_OFFSET


def _cpu_trace(pkw, *, absorbing: bool) -> np.ndarray:
    """Pressure at the sensor voxel from the CPU PSTD solver, `NT` samples."""
    grid = pkw.Grid(N, N, N, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_velocity_mask(_voxel_mask(SOURCE_IX), ux=_velocity_pulse(), mode="additive_no_correction")
    sensor = pkw.Sensor.from_mask(_voxel_mask(SENSOR_IX).astype(bool))
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML)
    sim.set_pml_inside(True)
    if absorbing:
        sim.set_alpha_coeff(ALPHA_COEFF_DB)
        sim.set_alpha_power(1.0)
        sim.set_power_law_dispersion(pkw.PowerLawDispersion.Omitted)
    return np.asarray(sim.run(time_steps=NT, dt=DT).sensor_data, dtype=np.float64).ravel()


def _session_trace(pkw, *, dispersion, absorbing: bool = True) -> np.ndarray:
    grid = pkw.Grid(N, N, N, DX, DX, DX)
    kwargs = {} if dispersion is None else {"power_law_dispersion": dispersion}
    try:
        session = pkw.GpuPstdSession(
            grid,
            np.full((N, N, N), C0),
            np.full((N, N, N), RHO0),
            dt=DT,
            time_steps=NT,
            absorption=np.full((N, N, N), ALPHA_COEFF_DB) if absorbing else None,
            pml_size_xyz=(PML, PML, PML),
            alpha_power=1.0,
            **kwargs,
        )
    except RuntimeError as exc:
        pytest.skip(f"GPU PSTD unavailable: {exc}")
    # One mask holds the source and the sensor voxel; the sensor voxel's
    # velocity row is zero, so only the first voxel drives the field. Voxels
    # are ordered by ascending x in both the source rows and the sensor rows.
    mask = _voxel_mask(SOURCE_IX) + _voxel_mask(SENSOR_IX)
    signals = np.vstack([_velocity_pulse(), np.zeros(NT)])
    session.set_source_sensor(mask, signals)
    session.disable_source_correction()
    traces = np.asarray(session.run_scan_line_cached(), dtype=np.float64)
    assert traces.shape == (2, NT)
    return traces[1]


def _normalised_by_lossless_peak(trace: np.ndarray, lossless: np.ndarray) -> np.ndarray:
    """Divide a trace by the peak of its own provider's lossless trace.

    The CPU solver and the session scale a velocity-source amplitude by
    different constants (their lossless traces differ by a fixed ratio,
    1.667 at this grid and time step). Linear PSTD is linear in the source
    amplitude, so normalising each provider by its own lossless peak removes
    the convention and leaves the absorption.
    """
    return trace / np.max(np.abs(lossless))


@pytest.mark.timeout(120)
def test_session_rejects_a_causal_unit_exponent_with_value_error():
    pkw = _pykwavers()
    grid = pkw.Grid(N, N, N, DX, DX, DX)
    try:
        with pytest.raises(ValueError, match="PowerLawDispersion::Omitted"):
            pkw.GpuPstdSession(
                grid,
                np.full((N, N, N), C0),
                np.full((N, N, N), RHO0),
                dt=DT,
                time_steps=NT,
                absorption=np.full((N, N, N), ALPHA_COEFF_DB),
                alpha_power=1.0,
                power_law_dispersion=pkw.PowerLawDispersion.Causal,
            )
    except RuntimeError as exc:
        pytest.skip(f"GPU PSTD unavailable: {exc}")


@pytest.mark.timeout(120)
def test_unit_exponent_without_dispersion_matches_the_cpu_solver():
    pkw = _pykwavers()
    omitted = pkw.PowerLawDispersion.Omitted
    gpu = _session_trace(pkw, dispersion=omitted)
    gpu_lossless = _session_trace(pkw, dispersion=omitted, absorbing=False)
    cpu = _cpu_trace(pkw, absorbing=True)
    cpu_lossless = _cpu_trace(pkw, absorbing=False)

    assert gpu.shape == cpu.shape == (NT,)
    gpu = _normalised_by_lossless_peak(gpu, gpu_lossless)
    cpu = _normalised_by_lossless_peak(cpu, cpu_lossless)
    cpu_unabsorbed = _normalised_by_lossless_peak(cpu_lossless, cpu_lossless)

    # The comparison sees the absorption operators ...
    assert np.max(np.abs(cpu_unabsorbed - cpu)) > ABSORPTION_VISIBILITY
    # ... and the session's operators produce the CPU solver's absorption.
    assert np.max(np.abs(cpu - gpu)) < PARITY_BOUND


def _cpu_simulation(pkw):
    mask = _voxel_mask(SOURCE_IX)
    grid = pkw.Grid(N, N, N, DX, DX, DX)
    medium = pkw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    source = pkw.Source.from_velocity_mask(mask, ux=_velocity_pulse(), mode="additive_no_correction")
    sensor = pkw.Sensor.from_mask(_voxel_mask(SENSOR_IX).astype(bool))
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML)
    sim.set_alpha_coeff(ALPHA_COEFF_DB)
    sim.set_alpha_power(1.0)
    return sim


def test_dispersion_is_causal_by_default_and_follows_the_setters():
    pkw = _pykwavers()
    sim = _cpu_simulation(pkw)
    assert sim.power_law_dispersion == pkw.PowerLawDispersion.Causal
    sim.set_power_law_dispersion(pkw.PowerLawDispersion.Omitted)
    assert sim.power_law_dispersion == pkw.PowerLawDispersion.Omitted
    config = pkw.NonlinearConfig().with_power_law_dispersion(pkw.PowerLawDispersion.Omitted)
    assert "power_law_dispersion=Omitted" in repr(config)
    sim.set_power_law_dispersion(pkw.PowerLawDispersion.Causal)
    sim.set_nonlinear_config(config)
    assert sim.power_law_dispersion == pkw.PowerLawDispersion.Omitted


@pytest.mark.timeout(120)
def test_cpu_solver_rejects_a_causal_unit_exponent_and_accepts_it_omitted():
    pkw = _pykwavers()
    sim = _cpu_simulation(pkw)
    with pytest.raises(Exception, match="PowerLawDispersion::Omitted"):
        sim.run(time_steps=NT, dt=DT)
    sim.set_power_law_dispersion(pkw.PowerLawDispersion.Omitted)
    trace = np.asarray(sim.run(time_steps=NT, dt=DT).sensor_data).ravel()
    assert trace.shape == (NT,)
    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
