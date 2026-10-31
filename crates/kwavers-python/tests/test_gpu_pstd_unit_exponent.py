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
# max |cpu - gpu| over the direct sensor traces: both providers scale an
# additive velocity source through the one shared 2 c0 dt / dx quotient, so
# no normalisation may hide a missing scale. The Rust parity test bounds f32
# PSTD against the f64 CPU path at gamma_(100 x 216) < 1.3e-3 for 100 steps
# of a 64^3 grid and asserts 2e-3; this 32^3 grid runs shorter transforms
# for fewer steps, so the same assertion bound applies.
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


def _cpu_trace(pkw, *, absorbing: bool, dt: float = DT) -> np.ndarray:
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
    return np.asarray(sim.run(time_steps=NT, dt=dt).sensor_data, dtype=np.float64).ravel()


def _session_trace(pkw, *, dispersion, absorbing: bool = True, dt: float = DT) -> np.ndarray:
    grid = pkw.Grid(N, N, N, DX, DX, DX)
    kwargs = {} if dispersion is None else {"power_law_dispersion": dispersion}
    try:
        session = pkw.GpuPstdSession(
            grid,
            np.full((N, N, N), C0),
            np.full((N, N, N), RHO0),
            dt=dt,
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


@pytest.mark.timeout(300)
@pytest.mark.parametrize("cfl", [0.2, 0.3])
def test_unit_exponent_without_dispersion_matches_the_cpu_solver(cfl):
    """The GPU session and the CPU solver agree on one sensor trace, directly.

    Both providers route an additive velocity source through the one shared
    `2 c0 dt / dx` quotient (`velocity_source_scale` in
    `crates/kwavers-source/src/grid_source.rs`), so the traces agree with no
    normalisation. A per-provider normalisation would hide a missing scale on
    either side; the direct comparison is the velocity-scale proof. The run
    is repeated at CFL 0.2 and 0.3 so the quotient is exercised away from a
    single timestep.
    """
    pkw = _pykwavers()
    dt = cfl * DX / C0
    omitted = pkw.PowerLawDispersion.Omitted
    gpu = _session_trace(pkw, dispersion=omitted, dt=dt)
    cpu = _cpu_trace(pkw, absorbing=True, dt=dt)
    cpu_lossless = _cpu_trace(pkw, absorbing=False, dt=dt)

    assert gpu.shape == cpu.shape == (NT,)
    # The comparison sees the absorption operators: same-provider relative
    # difference, so it cannot hide a cross-provider scale mismatch ...
    visibility = np.max(np.abs(cpu_lossless - cpu)) / np.max(np.abs(cpu_lossless))
    assert visibility > ABSORPTION_VISIBILITY
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
    with pytest.raises(RuntimeError, match="PowerLawDispersion::Omitted"):
        sim.run(time_steps=NT, dt=DT)
    sim.set_power_law_dispersion(pkw.PowerLawDispersion.Omitted)
    trace = np.asarray(sim.run(time_steps=NT, dt=DT).sensor_data).ravel()
    assert trace.shape == (NT,)
    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0.0


def _run_with_medium_exponent(pkw, dispersion, *, medium_exponent=1.0, alpha_power=1.5):
    """`run()` with the coefficient and exponent supplied by the medium."""
    mask = _voxel_mask(SOURCE_IX)
    grid = pkw.Grid(N, N, N, DX, DX, DX)
    medium = pkw.Medium.homogeneous(
        sound_speed=C0, density=RHO0, absorption=ALPHA_COEFF_DB, alpha_power=medium_exponent
    )
    source = pkw.Source.from_velocity_mask(mask, ux=_velocity_pulse(), mode="additive_no_correction")
    sensor = pkw.Sensor.from_mask(_voxel_mask(SENSOR_IX).astype(bool))
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML)
    sim.set_alpha_power(alpha_power)
    sim.set_power_law_dispersion(dispersion)
    return np.asarray(sim.run(time_steps=NT, dt=DT).sensor_data, dtype=np.float64).ravel()


def _run_with_explicit_coefficient(pkw, dispersion, exponent):
    """`run()` with the coefficient and exponent set on the simulation."""
    sim = _cpu_simulation(pkw)
    sim.set_alpha_power(exponent)
    sim.set_power_law_dispersion(dispersion)
    return np.asarray(sim.run(time_steps=NT, dt=DT).sensor_data, dtype=np.float64).ravel()


@pytest.mark.timeout(120)
def test_run_keeps_the_dispersion_when_the_medium_supplies_the_absorption():
    pkw = _pykwavers()
    omitted, causal = pkw.PowerLawDispersion.Omitted, pkw.PowerLawDispersion.Causal

    # A medium with y = 1 and Omitted runs as y = 1: the explicit-coefficient
    # path is the same solve with the coefficient given by the caller.
    from_medium = _run_with_medium_exponent(pkw, omitted)
    explicit_unit = _run_with_explicit_coefficient(pkw, omitted, 1.0)
    np.testing.assert_allclose(from_medium, explicit_unit, rtol=0.0, atol=1e-12 * np.max(np.abs(explicit_unit)))

    # Under Causal the medium's degenerate y = 1 is replaced by the configured
    # alpha_power (1.5), so the result is the y = 1.5 solve, not the y = 1 one.
    from_medium_causal = _run_with_medium_exponent(pkw, causal)
    explicit_replaced = _run_with_explicit_coefficient(pkw, causal, 1.5)
    np.testing.assert_allclose(
        from_medium_causal, explicit_replaced, rtol=0.0, atol=1e-12 * np.max(np.abs(explicit_replaced))
    )
    assert np.max(np.abs(from_medium_causal - from_medium)) > ABSORPTION_VISIBILITY * np.max(np.abs(from_medium))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
