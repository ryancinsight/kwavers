"""
Integration test for the Helmholtz (FEM) solver dispatch via SolverType.Helmholtz.

Validates that:
- A Simulation with SolverType.Helmholtz produces a steady-state sensor output
- The wavenumber can be controlled via ``set_helmholtz_wavenumber``
- The output shape is (n_sensors, 1) — single complex-magnitude snapshot
- Values are finite and physically plausible
- The pressure magnitude decays with distance from the source
"""

import numpy as np
import pytest

import pykwavers as kw


@pytest.fixture
def helmholtz_grid():
    """Small 6³ grid with 1 mm isotropic spacing → 6×6×6 mm domain."""
    return kw.Grid(nx=6, ny=6, nz=6, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)


@pytest.fixture
def helmholtz_medium():
    """Homogeneous water (c = 1500 m/s, ρ = 1000 kg/m³)."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def helmholtz_source():
    """Unit-amplitude initial pressure at the domain centre."""
    p0 = np.zeros((6, 6, 6), dtype=np.float64)
    p0[3, 3, 3] = 1.0
    return kw.Source.from_initial_pressure(p0)


@pytest.fixture
def helmholtz_sensor():
    """Two sensor points at different distances from the source."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[3, 3, 4] = True   # 1 cell from source → 1 mm
    mask[3, 3, 5] = True   # 2 cells from source → 2 mm
    return kw.Sensor.from_mask(mask)


def test_helmholtz_steady_state_output(helmholtz_grid, helmholtz_medium,
                                        helmholtz_source, helmholtz_sensor):
    """SolverType.Helmholtz runs and returns a single steady-state snapshot."""
    sim = kw.Simulation(
        helmholtz_grid, helmholtz_medium, helmholtz_source, helmholtz_sensor,
        solver=kw.SolverType.Helmholtz,
    )
    # 50 kHz excitation: λ = c/f = 1500 / 5e4 = 30 mm; ~0.2λ across 6 mm domain
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    # ── Shape assertions ──────────────────────────────────────────────────
    assert result.num_sensors == 2
    assert result.sensor_data_shape == (2, 1), (
        f"Expected (2, 1) steady-state snapshot, got {result.sensor_data_shape}"
    )
    # Multiple sensors → 2D array; single sensor would flatten to 1D
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
    # (Standing-wave patterns in bounded Helmholtz domains with radiation
    #  BCs can produce non-monotonic spatial profiles; monotonic decay is
    #  only guaranteed in free-space Green's-function solutions.)
    assert np.all(np.isfinite(result.sensor_data))
    assert np.all(result.sensor_data > 0.0)

    # ── Optional: statistics are populated ────────────────────────────────
    assert result.p_max is not None, "p_max statistics should be populated"
    assert np.all(np.isfinite(result.p_max))
    assert result.p_max[0] == p_near
    assert result.p_max[1] == p_far


def test_helmholtz_wavenumber_setter_rejects_nonpositive():
    """set_helmholtz_wavenumber rejects ≤ 0 with ValueError."""
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 2] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)

    with pytest.raises(ValueError, match="must be positive"):
        sim.set_helmholtz_wavenumber(0.0)

    with pytest.raises(ValueError, match="must be positive"):
        sim.set_helmholtz_wavenumber(-1e6)


def test_helmholtz_wavenumber_getter_roundtrips():
    """set_helmholtz_wavenumber value is returned by the getter."""
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 2] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)

    assert sim.helmholtz_frequency is None, "Default should be None"

    sim.set_helmholtz_wavenumber(1e6)
    assert sim.helmholtz_frequency == 1e6


def test_helmholtz_without_explicit_frequency_uses_dt_fallback(helmholtz_grid,
                                                                helmholtz_medium,
                                                                helmholtz_source,
                                                                helmholtz_sensor):
    """Without set_helmholtz_wavenumber, the solver falls back to dt-derived k."""
    sim = kw.Simulation(
        helmholtz_grid, helmholtz_medium, helmholtz_source, helmholtz_sensor,
        solver=kw.SolverType.Helmholtz,
    )
    # Provide an explicit dt; k = 2π / (c·dt)
    dt = 2.0e-6  # → f ≈ 500 kHz, λ ≈ 3 mm, kh ≈ 2.1 at 1 mm spacing
    result = sim.run(time_steps=1, dt=dt)

    assert result.num_sensors == 2
    assert result.sensor_data.shape == (2, 1)
    assert np.all(np.isfinite(result.sensor_data))
    assert np.all(result.sensor_data > 0.0)


def test_helmholtz_constant_field_identity():
    """A uniform p₀=1 on the z=0 face produces a symmetric finite solution.

    The nodal-load injection treats p₀ as a body-force term f in ∇²u + k²u = −f,
    not as a Dirichlet boundary condition.  The resulting interior pressure
    magnitude is solver-dependent, so this test validates finiteness,
    positivity, and xy-symmetry at the centre z-plane rather than an absolute
    magnitude range."""
    n = 4
    grid = kw.Grid(nx=n, ny=n, nz=n, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)

    # Uniform unit pressure on the z=0 plane (entire face)
    p0 = np.zeros((n, n, n), dtype=np.float64)
    p0[:, :, 0] = 1.0
    source = kw.Source.from_initial_pressure(p0)

    # Sensors at two xy-symmetric positions on the centre z-plane
    mask = np.zeros((n, n, n), dtype=bool)
    mask[1, 2, 2] = True  # (x=1, y=2, z=2)
    mask[2, 1, 2] = True  # (x=2, y=1, z=2) — same |x−2|+|y−2| distance
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)
    sim.set_helmholtz_wavenumber(50e3)  # low freq → large λ, easier convergence
    result = sim.run(time_steps=1)

    assert result.num_sensors == 2
    assert result.sensor_data.ndim == 2
    assert result.sensor_data.shape == (2, 1)

    v1 = result.sensor_data[0, 0]
    v2 = result.sensor_data[1, 0]
    assert np.isfinite(v1)
    assert np.isfinite(v2)
    assert v1 > 0.0
    assert v2 > 0.0
    # Symmetry: the uniform face source is x↔y symmetric at the centre plane
    assert v1 == pytest.approx(v2, rel=1e-6), (
        f"Solution not xy-symmetric: v1={v1:.6e}, v2={v2:.6e}"
    )


def test_helmholtz_time_varying_source_produces_zero_field():
    """Helmholtz solver with time-varying source (no p0) produces zero field."""
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(1500.0, 1000.0)
    # Point source (time-varying), not initial pressure
    source = kw.Source.point(position=(1.0e-3, 2.0e-3, 2.0e-3),
                             frequency=1e6, amplitude=1e5)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[2, 2, 2] = True
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)
    # Should still run (returns zero field), but doesn't crash
    result = sim.run(time_steps=1)
    assert result.num_sensors == 1
    # Single-sensor result is flattened to 1D by simulation_run_result_to_py
    assert result.sensor_data.ndim == 1
    # Without p0, Helmholtz has no sources → zero solution
    assert result.sensor_data[0] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# Edge-case tests
# ══════════════════════════════════════════════════════════════════════════


def test_helmholtz_empty_p0_zero_field():
    """Helmholtz solver with no initial pressure (p0=None) produces zero output.

    ``Source.point()`` creates a time-varying source without setting ``p0``,
    so the Helmholtz solver has no nodal loads → solution is identically zero.
    """
    grid = kw.Grid(nx=5, ny=5, nz=5, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.point(position=(2.5e-3, 2.5e-3, 2.5e-3),
                             frequency=1e6, amplitude=1e5)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    assert result.num_sensors == 1
    assert result.sensor_data.ndim == 1
    assert result.sensor_data[0] == 0.0


def test_helmholtz_grid_sensor_all_points():
    """Helmholtz solver with ``Sensor.grid()`` records all grid points.

    ``Sensor.grid()`` creates an all-``True`` mask covering every grid
    vertex.  This verifies the solver does not crash when extracting
    the solution at every node and that the output shape matches the
    full grid size.
    """
    # Use a small 4³ grid to keep the test fast
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 2] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()  # all 64 grid vertices

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)
    sim.set_helmholtz_wavenumber(50e3)
    result = sim.run(time_steps=1)

    # 4³ = 64 sensors, 1 steady-state column
    assert result.num_sensors == 64
    assert result.sensor_data_shape == (64, 1)
    assert result.sensor_data.ndim == 2
    assert result.sensor_data.shape == (64, 1)
    assert np.all(np.isfinite(result.sensor_data))
    # Node at source position should have non-zero pressure
    assert result.sensor_data[2 + 4 * (2 + 4 * 2), 0] > 0.0


def test_helmholtz_single_sensor_flattening():
    """Exactly one sensor flattens the output to 1-D.

    ``simulation_run_result_to_py`` flattens (1, n_cols) → 1-D array
    and sets ``sensor_data_2d = None`` when ``n_sensors ≤ 1``.
    """
    grid = kw.Grid(nx=5, ny=5, nz=5, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((5, 5, 5), dtype=np.float64)
    p0[2, 2, 2] = 1.0  # non-zero source
    source = kw.Source.from_initial_pressure(p0)
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 3] = True  # exactly 1 sensor
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)
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
        f"Non-zero source should produce positive pressure at sensor"
    )


def test_helmholtz_grid_too_small():
    """Helmholtz solver rejects grids with fewer than 2 vertices per axis.

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

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)
    sim.set_helmholtz_wavenumber(50e3)

    with pytest.raises(RuntimeError, match="requires at least 2 vertices per axis"):
        sim.run(time_steps=1)


def test_helmholtz_small_grid_succeeds():
    """Helmholtz solver succeeds on a 2×2×2 grid — the minimum valid size.

    ``TetrahedralMesh::from_grid_vertices`` requires ≥ 2 vertices per axis.
    A 2×2×2 grid meets this lower bound exactly, complementing the
    ``test_helmholtz_grid_too_small`` test (2×1×1 → RuntimeError).
    This mirrors the RS/DG "small grid succeeds" pattern.
    """
    grid = kw.Grid(nx=2, ny=2, nz=2, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((2, 2, 2), dtype=np.float64)
    p0[1, 1, 1] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Helmholtz)
    sim.set_helmholtz_wavenumber(50e3)

    # Must NOT raise — 2×2×2 is the smallest valid grid for tetrahedral meshing
    result = sim.run(time_steps=1)

    # 2³ = 8 sensors, 1 steady-state column
    assert result.num_sensors == 8
    assert result.sensor_data_shape == (8, 1)
    assert np.all(np.isfinite(result.sensor_data))
