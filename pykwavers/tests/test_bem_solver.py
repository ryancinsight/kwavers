"""
Integration test for the BEM solver dispatch via SolverType.BEM.

Validates that:
- A Simulation with SolverType.BEM produces a steady-state sensor output
- The wavenumber can be controlled via ``set_helmholtz_wavenumber``
- The output shape is (n_sensors, 1) — single complex-magnitude snapshot
- Values are finite and physically plausible
- The solver degrades gracefully without p0 (zero field)
"""

import numpy as np
import pytest

import pykwavers as kw


@pytest.fixture
def bem_grid():
    """Small 6³ grid with 1 mm isotropic spacing → 6×6×6 mm domain."""
    return kw.Grid(nx=6, ny=6, nz=6, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)


@pytest.fixture
def bem_medium():
    """Homogeneous water (c = 1500 m/s, ρ = 1000 kg/m³)."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def bem_source():
    """Initial pressure on the z=0 boundary face (required for BEM Dirichlet BCs).

    BEM only applies Dirichlet BCs at boundary nodes, so the source must
    reside on one of the 6 bounding faces.  A unit-amplitude patch is placed
    at the centre of the z=0 face.
    """
    p0 = np.zeros((6, 6, 6), dtype=np.float64)
    p0[3, 3, 0] = 1.0  # centre of z=0 face
    return kw.Source.from_initial_pressure(p0)


@pytest.fixture
def bem_sensor():
    """Two sensor points in the interior at different distances from the source."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[3, 3, 2] = True  # 2 cells from z=0 face → 2 mm
    mask[3, 3, 4] = True  # 4 cells from z=0 face → 4 mm
    return kw.Sensor.from_mask(mask)


def test_bem_steady_state_output(bem_grid, bem_medium, bem_source, bem_sensor):
    """SolverType.BEM runs and returns a single steady-state snapshot."""
    sim = kw.Simulation(
        bem_grid, bem_medium, bem_source, bem_sensor,
        solver=kw.SolverType.BEM,
    )
    # 50 kHz excitation: λ = c/f = 1500 / 5e4 = 30 mm; ~0.2λ across 6 mm domain
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
    p_near = result.sensor_data[0, 0]
    p_far = result.sensor_data[1, 0]

    assert np.isfinite(p_near), f"Sensor 0 value not finite: {p_near}"
    assert np.isfinite(p_far), f"Sensor 1 value not finite: {p_far}"
    assert p_near > 0.0, f"Sensor 0 magnitude must be positive, got {p_near}"
    assert p_far > 0.0, f"Sensor 1 magnitude must be positive, got {p_far}"

    # ── Physics: both sensors register finite positive pressure ──────────
    # (BEM with radiation BCs models an open domain; the scattered field
    #  from a boundary Dirichlet source is non-zero throughout.)
    assert np.all(np.isfinite(result.sensor_data))
    assert np.all(result.sensor_data > 0.0)

    # ── Optional: statistics are populated ────────────────────────────────
    assert result.p_max is not None, "p_max statistics should be populated"
    assert np.all(np.isfinite(result.p_max))
    assert result.p_max[0] == p_near
    assert result.p_max[1] == p_far


def test_bem_wavenumber_setter_rejects_nonpositive():
    """set_helmholtz_wavenumber rejects ≤ 0 with ValueError (shared setter)."""
    grid = kw.Grid(nx=5, ny=5, nz=5, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((5, 5, 5), dtype=np.float64)
    p0[2, 2, 0] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)

    with pytest.raises(ValueError, match="must be positive"):
        sim.set_helmholtz_wavenumber(0.0)

    with pytest.raises(ValueError, match="must be positive"):
        sim.set_helmholtz_wavenumber(-1e6)


def test_bem_wavenumber_getter_roundtrips():
    """set_helmholtz_wavenumber value is returned by the getter (shared)."""
    grid = kw.Grid(nx=5, ny=5, nz=5, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((5, 5, 5), dtype=np.float64)
    p0[2, 2, 0] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)

    assert sim.helmholtz_frequency is None, "Default should be None"

    sim.set_helmholtz_wavenumber(1e6)
    assert sim.helmholtz_frequency == 1e6


def test_bem_without_explicit_frequency_uses_dt_fallback(bem_grid, bem_medium,
                                                          bem_source, bem_sensor):
    """Without set_helmholtz_wavenumber, BEM falls back to dt-derived k."""
    sim = kw.Simulation(
        bem_grid, bem_medium, bem_source, bem_sensor,
        solver=kw.SolverType.BEM,
    )
    # Provide an explicit dt; k = 2π / (c·dt)
    dt = 2.0e-6  # → f ≈ 500 kHz, λ ≈ 3 mm
    result = sim.run(time_steps=1, dt=dt)

    assert result.num_sensors == 2
    assert result.sensor_data.shape == (2, 1)
    assert np.all(np.isfinite(result.sensor_data))
    assert np.all(result.sensor_data > 0.0)


def test_bem_time_varying_source_produces_zero_field():
    """BEM solver with time-varying source (no p0) produces zero field."""
    grid = kw.Grid(nx=5, ny=5, nz=5, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(1500.0, 1000.0)
    # Point source (time-varying), not initial pressure — no boundary p0
    source = kw.Source.point(position=(1.0e-3, 2.0e-3, 2.0e-3),
                             frequency=1e6, amplitude=1e5)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)
    # Should still run (returns zero field), but doesn't crash
    result = sim.run(time_steps=1)
    assert result.num_sensors == 1
    # Single-sensor result is flattened to 1D by simulation_run_result_to_py
    assert result.sensor_data.ndim == 1
    # Without p0, BEM has no Dirichlet BCs → zero solution
    assert result.sensor_data[0] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# Edge-case tests
# ══════════════════════════════════════════════════════════════════════════


def test_bem_empty_p0_zero_field():
    """BEM solver with no initial pressure (p0=None) produces zero output.

    ``Source.point()`` creates a time-varying source without setting ``p0``,
    so the BEM solver has no Dirichlet BCs → all boundary nodes get
    radiation BCs → zero scattered field.
    """
    grid = kw.Grid(nx=5, ny=5, nz=5, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.point(position=(2.5e-3, 2.5e-3, 2.5e-3),
                             frequency=1e6, amplitude=1e5)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    assert result.num_sensors == 1
    assert result.sensor_data.ndim == 1
    assert result.sensor_data[0] == 0.0


def test_bem_grid_sensor_all_points():
    """BEM solver with ``Sensor.grid()`` records all grid points.

    ``Sensor.grid()`` creates an all-``True`` mask covering every grid
    vertex.  This verifies the BEM solver does not crash when computing
    the scattered field at every node and that the output shape matches
    the full grid size.
    """
    # Use a small 4³ grid to keep the test fast
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 0] = 1.0  # boundary face source
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()  # all 64 grid vertices

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    # 4³ = 64 sensors, 1 steady-state column
    assert result.num_sensors == 64
    assert result.sensor_data_shape == (64, 1)
    assert result.sensor_data.ndim == 2
    assert result.sensor_data.shape == (64, 1)
    assert np.all(np.isfinite(result.sensor_data))
    # BEM computed field at interior points should be finite
    mid_idx = 2 + 4 * (2 + 4 * 2)  # (2,2,2) in row-major
    assert np.isfinite(result.sensor_data[mid_idx, 0])


def test_bem_single_sensor_flattening():
    """Exactly one BEM sensor flattens the output to 1-D.

    ``simulation_run_result_to_py`` flattens (1, n_cols) → 1-D array
    and sets ``sensor_data_2d = None`` when ``n_sensors ≤ 1``.
    """
    grid = kw.Grid(nx=5, ny=5, nz=5, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((5, 5, 5), dtype=np.float64)
    p0[2, 2, 0] = 1.0  # boundary face source
    source = kw.Source.from_initial_pressure(p0)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 3] = True  # exactly 1 interior sensor
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)
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
        f"Non-zero boundary source should produce positive pressure at sensor"
    )


def test_bem_grid_too_small():
    """BEM solver rejects grids with fewer than 2 vertices per axis.

    ``TetrahedralMesh::from_grid_vertices`` requires at least 2 vertices
    along each axis; a 2×1×1 grid triggers an ``InvalidInput`` error
    propagated as a Python ``RuntimeError``.
    """
    grid = kw.Grid(nx=2, ny=1, nz=1, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((2, 1, 1), dtype=np.float64)
    p0[0, 0, 0] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)
    sim.set_helmholtz_wavenumber(50e3)

    with pytest.raises(RuntimeError, match="requires at least 2 vertices per axis"):
        sim.run(time_steps=1)


def test_bem_small_grid_succeeds():
    """BEM solver succeeds on a 2×2×2 grid — the minimum valid size.

    ``TetrahedralMesh::from_grid_vertices`` requires ≥ 2 vertices per axis.
    A 2×2×2 grid meets this lower bound exactly, complementing the
    ``test_bem_grid_too_small`` test (2×1×1 → RuntimeError).
    This mirrors the RS/DG "small grid succeeds" pattern.
    """
    grid = kw.Grid(nx=2, ny=2, nz=2, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    # BEM requires boundary source — place it on a face
    p0 = np.zeros((2, 2, 2), dtype=np.float64)
    p0[1, 1, 0] = 1.0  # centre of z=0 face
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.BEM)
    sim.set_helmholtz_wavenumber(50e3)

    # Must NOT raise — 2×2×2 is the smallest valid grid for tetrahedral meshing
    result = sim.run(time_steps=1)

    # 2³ = 8 sensors, 1 steady-state column
    assert result.num_sensors == 8
    assert result.sensor_data_shape == (8, 1)
    assert np.all(np.isfinite(result.sensor_data))
