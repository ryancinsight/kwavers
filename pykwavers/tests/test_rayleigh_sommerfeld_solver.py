"""
Integration test for the Rayleigh-Sommerfeld (FNM) solver dispatch via
``SolverType.RayleighSommerfeld``.

Validates that:
- A ``Simulation`` with ``SolverType.RayleighSommerfeld`` and a transducer
  produces a steady-state sensor output
- The solver is transducer-driven: p₀ is ignored, and the transducer's
  velocity distribution drives the angular-spectrum propagation
- The output shape is ``(n_sensors, 1)`` — single pressure-magnitude snapshot
- Values are finite and physically plausible
- Single-sensor output is flattened to 1-D by ``simulation_run_result_to_py``
- A ``RuntimeError`` is raised when no transducer is attached
"""

import numpy as np
import pytest

import pykwavers as kw


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rs_grid():
    """6³ grid with 1 mm isotropic spacing → 6×6×6 mm domain."""
    return kw.Grid(nx=6, ny=6, nz=6, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)


@pytest.fixture
def rs_medium():
    """Homogeneous water (c = 1500 m/s, ρ = 1000 kg/m³)."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def rs_transducer():
    """8-element linear array with 0.5 mm pitch, 0.3 mm element width.

    Total aperture = 4.0 mm, operating at 50 kHz.
    """
    return kw.TransducerArray2D(
        number_elements=8,
        element_width=0.3e-3,
        element_length=6.0e-3,
        element_spacing=0.5e-3,
        sound_speed=1500.0,
        frequency=50e3,
    )


@pytest.fixture
def rs_sensor():
    """Two sensor points on the same z-plane at different x-positions."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2, 3, 4] = True  # x=2 mm (left of centre)
    mask[4, 3, 4] = True  # x=4 mm (right of centre)
    return kw.Sensor.from_mask(mask)


# ══════════════════════════════════════════════════════════════════════════
# Steady-state output
# ══════════════════════════════════════════════════════════════════════════


def test_rs_steady_state_output(rs_grid, rs_medium, rs_transducer, rs_sensor):
    """SolverType.RayleighSommerfeld runs and returns a steady-state snapshot."""
    sim = kw.Simulation(
        rs_grid, rs_medium, rs_transducer, rs_sensor,
        solver=kw.SolverType.RayleighSommerfeld,
    )
    # 50 kHz — same as the transducer's native frequency
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    # ── Shape assertions ──────────────────────────────────────────────────
    assert result.num_sensors == 2
    assert result.sensor_data_shape == (2, 1), (
        f"Expected (2, 1) steady-state snapshot, got {result.sensor_data_shape}"
    )
    assert result.sensor_data.ndim == 2
    assert result.sensor_data.shape == (2, 1)

    # ── Value assertions ──────────────────────────────────────────────────
    p_left = result.sensor_data[0, 0]
    p_right = result.sensor_data[1, 0]

    assert np.isfinite(p_left), f"Sensor 0 value not finite: {p_left}"
    assert np.isfinite(p_right), f"Sensor 1 value not finite: {p_right}"
    assert p_left > 0.0, f"Sensor 0 magnitude must be positive, got {p_left}"
    assert p_right > 0.0, f"Sensor 1 magnitude must be positive, got {p_right}"

    # Both sensors see the transducer field (8-element array centred at origin)
    assert np.all(np.isfinite(result.sensor_data))
    assert np.all(result.sensor_data > 0.0)

    # ── Statistics are populated ──────────────────────────────────────────
    assert result.p_max is not None, "p_max statistics should be populated"
    assert np.all(np.isfinite(result.p_max))
    assert result.p_max[0] == p_left
    assert result.p_max[1] == p_right


def test_rs_without_explicit_frequency_uses_transducer_fallback(
    rs_grid, rs_medium, rs_transducer, rs_sensor,
):
    """Without ``set_helmholtz_wavenumber``, RS falls back to transducer frequency.

    Unlike Helmholtz/BEM which fall back to ``dt``, the RS solver prefers
    the transducer's own ``frequency`` property before trying ``1/dt``.
    """
    sim = kw.Simulation(
        rs_grid, rs_medium, rs_transducer, rs_sensor,
        solver=kw.SolverType.RayleighSommerfeld,
    )
    # No set_helmholtz_wavenumber call — falls back to transducer.frequency (50 kHz)
    result = sim.run(time_steps=1)

    assert result.num_sensors == 2
    assert result.sensor_data.shape == (2, 1)
    assert np.all(np.isfinite(result.sensor_data))
    assert np.all(result.sensor_data > 0.0)


# ══════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════


def test_rs_empty_p0_still_produces_output(rs_grid, rs_medium, rs_transducer):
    """RS solver ignores p₀ — a transducer alone produces valid output.

    The Rayleigh-Sommerfeld solver is transducer-driven; ``_grid_source``
    is never inspected.  Passing ``Source.point()`` (which sets p₀=None)
    alongside a transducer should still compute the transducer's radiated
    pressure field without error.
    """
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[3, 3, 4] = True  # 1 sensor at centre of evaluation plane
    sensor = kw.Sensor.from_mask(mask)

    # Pass a list: [point source (p₀=None), transducer]
    point_source = kw.Source.point(
        position=(3.0e-3, 3.0e-3, 2.0e-3),
        frequency=1e6, amplitude=1e5,
    )

    sim = kw.Simulation(
        rs_grid, rs_medium, [point_source, rs_transducer], sensor,
        solver=kw.SolverType.RayleighSommerfeld,
    )
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    # The RS solver uses only the transducer — p₀ is ignored.
    # Output should be non-zero because the transducer is active.
    assert result.num_sensors == 1
    assert result.sensor_data.ndim == 1
    assert np.isfinite(result.sensor_data[0])
    assert result.sensor_data[0] > 0.0, (
        "RS solver is transducer-driven; p₀=None should not prevent output"
    )


def test_rs_single_sensor_flattening(rs_grid, rs_medium, rs_transducer):
    """Exactly one RS sensor flattens the output to 1-D.

    ``simulation_run_result_to_py`` flattens ``(1, n_cols)`` → 1-D array
    and sets ``sensor_data_2d = None`` when ``n_sensors ≤ 1``.
    """
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[3, 3, 4] = True  # exactly 1 sensor
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(
        rs_grid, rs_medium, rs_transducer, sensor,
        solver=kw.SolverType.RayleighSommerfeld,
    )
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    # Single-sensor flattening
    assert result.num_sensors == 1
    assert result.sensor_data_shape == (1, 1)
    assert result.sensor_data.ndim == 1, (
        f"Expected 1-D flattened array, got ndim={result.sensor_data.ndim}"
    )
    assert np.isfinite(result.sensor_data[0])
    assert result.sensor_data[0] > 0.0, (
        "Transducer should produce positive pressure at sensor"
    )


def test_rs_grid_sensor_all_points(rs_grid, rs_medium, rs_transducer):
    """RS solver with ``Sensor.grid()`` records all grid points.

    ``Sensor.grid()`` creates an all-``True`` mask covering every grid
    vertex.  This verifies the RS solver does not crash when mapping
    every node to its nearest FNM pixel column and that the output
    shape matches the full grid size.
    """
    # Use a small 4³ grid to keep the test fast
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)

    # 4-element transducer for a 4 mm aperture matching the grid width
    transducer = kw.TransducerArray2D(
        number_elements=4,
        element_width=0.5e-3,
        element_length=4.0e-3,
        element_spacing=1.0e-3,
        sound_speed=1500.0,
        frequency=50e3,
    )
    sensor = kw.Sensor.grid()  # all 64 grid vertices

    sim = kw.Simulation(
        grid, medium, transducer, sensor,
        solver=kw.SolverType.RayleighSommerfeld,
    )
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    # 4³ = 64 sensors, 1 steady-state column
    assert result.num_sensors == 64
    assert result.sensor_data_shape == (64, 1)
    assert result.sensor_data.ndim == 2
    assert result.sensor_data.shape == (64, 1)
    assert np.all(np.isfinite(result.sensor_data))
    # At least some sensor positions should see the transducer field
    assert np.any(result.sensor_data > 0.0), (
        "At least one grid vertex should register positive pressure"
    )


def test_rs_small_grid_succeeds():
    """RS solver tolerates grids too small for tetrahedral mesh generation.

    Unlike Helmholtz and BEM — which require at least 2 vertices per axis
    for ``TetrahedralMesh::from_grid_vertices`` — the RS solver does not
    generate a mesh.  A 2×1×1 grid (which fails Helmholtz/BEM with
    ``RuntimeError``) completes normally with RS.
    """
    grid = kw.Grid(nx=2, ny=1, nz=1, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    transducer = kw.TransducerArray2D(
        number_elements=2,
        element_width=0.3e-3,
        element_length=1.0e-3,
        element_spacing=0.5e-3,
        sound_speed=1500.0,
        frequency=50e3,
    )
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(
        grid, medium, transducer, sensor,
        solver=kw.SolverType.RayleighSommerfeld,
    )
    sim.set_helmholtz_wavenumber(50e3)

    # Must NOT raise — RS doesn't generate a tetrahedral mesh
    result = sim.run(time_steps=1)

    # 2×1×1 = 2 sensors
    assert result.num_sensors == 2
    assert result.sensor_data_shape == (2, 1)
    assert np.all(np.isfinite(result.sensor_data))


def test_rs_no_transducer_error(rs_grid, rs_medium):
    """RS solver raises RuntimeError when no transducer is attached.

    The Rayleigh-Sommerfeld solver is transducer-driven and requires at
    least one ``TransducerArray2D``.  Passing only a ``Source`` (p₀ alone)
    triggers an ``InvalidInput`` error propagated as a Python ``RuntimeError``.
    """
    p0 = np.zeros((6, 6, 6), dtype=np.float64)
    p0[3, 3, 3] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(
        rs_grid, rs_medium, source, sensor,
        solver=kw.SolverType.RayleighSommerfeld,
    )
    sim.set_helmholtz_wavenumber(50e3)

    with pytest.raises(RuntimeError, match="requires at least one TransducerArray2D"):
        sim.run(time_steps=1)
