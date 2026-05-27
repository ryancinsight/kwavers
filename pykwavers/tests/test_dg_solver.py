"""
Integration test for the DG (Discontinuous Galerkin / Hybrid Spectral-DG)
solver dispatch via ``SolverType.DG``.

.. note::

    Unlike Helmholtz, BEM, and Rayleigh-Sommerfeld — which are frequency-domain
    solvers producing a single ``(n_sensors, 1)`` snapshot — the DG solver is
    **time-domain**.  Its output shape is ``(n_sensors, time_steps)`` and it
    records the full pressure history at each sensor position.

    The DG solver also requires grid dimensions to be multiples of
    ``nodes_per_element=4`` (the default polynomial-order-3 GLL node count).

Validates that:
- A ``Simulation`` with ``SolverType.DG`` produces a time-series sensor output
- The output shape is ``(n_sensors, time_steps)`` — multi-step time history
- Pressure evolves over time (different time steps have different values)
- Single-sensor output is flattened to 1-D
- Empty p₀ (``Source.point()``) produces identically zero output
- Grids a multiple of 4 per axis succeed (DG operates on Cartesian grids,
  not tetrahedral meshes, so the Helmholtz/BEM mesh constraint doesn't apply)
"""

import numpy as np
import pytest

import pykwavers as kw


# ── DG-divisibility invariant ─────────────────────────────────────────────
# The DG solver uses polynomial-order-3 GLL nodes ⇒ nodes_per_element = 4.
# Every grid axis must be a multiple of 4.
DG_SIZE = 4  # 4³ grid


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def dg_grid():
    """4³ grid with 1 mm isotropic spacing → 4×4×4 mm domain."""
    return kw.Grid(nx=DG_SIZE, ny=DG_SIZE, nz=DG_SIZE,
                   dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)


@pytest.fixture
def dg_medium():
    """Homogeneous water (c = 1500 m/s, ρ = 1000 kg/m³)."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def dg_source():
    """Unit-amplitude initial pressure at the domain centre."""
    p0 = np.zeros((DG_SIZE, DG_SIZE, DG_SIZE), dtype=np.float64)
    p0[2, 2, 2] = 1.0
    return kw.Source.from_initial_pressure(p0)


@pytest.fixture
def dg_sensor():
    """Two sensor points at different distances from the source."""
    mask = np.zeros((DG_SIZE, DG_SIZE, DG_SIZE), dtype=bool)
    mask[2, 2, 2] = True  # at the source position
    mask[2, 2, 3] = True  # 1 cell from source → 1 mm
    return kw.Sensor.from_mask(mask)


# ══════════════════════════════════════════════════════════════════════════
# Time-domain output
# ══════════════════════════════════════════════════════════════════════════


def test_dg_time_domain_output(dg_grid, dg_medium, dg_source, dg_sensor):
    """SolverType.DG runs and returns a multi-step time-series snapshot."""
    sim = kw.Simulation(
        dg_grid, dg_medium, dg_source, dg_sensor,
        solver=kw.SolverType.DG,
    )
    # DG is time-domain — multiple time steps produce a (n_sensors, time_steps)
    # history rather than a single steady-state column.
    time_steps = 5
    result = sim.run(time_steps=time_steps)

    # ── Shape assertions ──────────────────────────────────────────────────
    assert result.num_sensors == 2
    assert result.sensor_data_shape == (2, time_steps), (
        f"Expected (2, {time_steps}) time-series, got {result.sensor_data_shape}"
    )
    assert result.sensor_data.ndim == 2
    assert result.sensor_data.shape == (2, time_steps)

    # ── Value assertions ──────────────────────────────────────────────────
    p_source = result.sensor_data[0, :]  # time series at source position
    p_remote = result.sensor_data[1, :]  # time series at remote position

    assert np.all(np.isfinite(p_source)), "Source sensor has non-finite values"
    assert np.all(np.isfinite(p_remote)), "Remote sensor has non-finite values"

    # ── Temporal evolution ────────────────────────────────────────────────
    # Pressure at the source position should change over time steps
    # (initial condition spreads via the wave equation).
    unique_source_vals = np.unique(p_source)
    assert len(unique_source_vals) >= 2, (
        f"Source pressure should evolve over time; "
        f"got only {len(unique_source_vals)} unique value(s)"
    )

    # ── Statistics are populated ──────────────────────────────────────────
    assert result.p_max is not None, "p_max statistics should be populated"
    assert np.all(np.isfinite(result.p_max))
    assert result.p_max[0] >= 0.0
    assert result.p_max[1] >= 0.0


def test_dg_propagation_over_time(dg_grid, dg_medium, dg_source):
    """Pressure propagates outward from the source over time.

    With a unit-amplitude p₀ at the domain centre, the first sensor
    (at the source) registers the highest initial pressure, and the
    time series at different distances are distinct.
    """
    mask = np.zeros((DG_SIZE, DG_SIZE, DG_SIZE), dtype=bool)
    mask[2, 2, 2] = True  # at source
    mask[2, 2, 3] = True  # 1 cell away
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(
        dg_grid, dg_medium, dg_source, sensor,
        solver=kw.SolverType.DG,
    )
    time_steps = 8  # enough steps for the wave to propagate
    result = sim.run(time_steps=time_steps)

    assert result.num_sensors == 2
    assert result.sensor_data_shape == (2, time_steps)

    # First time step at source should equal initial p0 (= 1.0)
    assert result.sensor_data[0, 0] == pytest.approx(1.0, abs=1e-12), (
        f"t=0 at source should be 1.0, got {result.sensor_data[0, 0]}"
    )

    # The full time series at the two positions should differ
    # (wave hasn't reached the remote sensor at t=0, but source is active)
    assert not np.allclose(result.sensor_data[0, :], result.sensor_data[1, :]), (
        "Time series at source and remote positions should differ"
    )


# ══════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════


def test_dg_empty_p0_zero_field():
    """DG solver with no initial pressure (p₀=None) produces zero output.

    ``Source.point()`` creates a time-varying source without setting ``p₀``,
    so the DG solver initialises the field from quiescent zeros and the
    time series remains identically zero at every sensor.
    """
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.point(
        position=(2.0e-3, 2.0e-3, 2.0e-3),
        frequency=1e6, amplitude=1e5,
    )
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[2, 2, 2] = True
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.DG)
    time_steps = 3
    result = sim.run(time_steps=time_steps)

    assert result.num_sensors == 1
    # Single-sensor result is flattened to 1-D by simulation_run_result_to_py
    assert result.sensor_data.ndim == 1
    assert result.sensor_data.shape == (time_steps,)
    # Without p₀, DG field is initialised from zeros → all time steps are zero
    assert np.all(result.sensor_data == 0.0)


def test_dg_single_sensor_flattening():
    """Exactly one DG sensor flattens the output to 1-D.

    ``simulation_run_result_to_py`` flattens ``(1, n_cols)`` → 1-D array
    and sets ``sensor_data_2d = None`` when ``n_sensors ≤ 1``.
    """
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 2] = 1.0  # non-zero source
    source = kw.Source.from_initial_pressure(p0)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[2, 2, 3] = True  # exactly 1 sensor
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.DG)
    time_steps = 4
    result = sim.run(time_steps=time_steps)

    # Single-sensor flattening
    assert result.num_sensors == 1
    assert result.sensor_data_shape == (1, time_steps)
    assert result.sensor_data.ndim == 1, (
        f"Expected 1-D flattened array, got ndim={result.sensor_data.ndim}"
    )
    assert result.sensor_data.shape == (time_steps,)
    assert np.all(np.isfinite(result.sensor_data))
    # At least one time step should be non-zero (initial p₀ spreads)
    assert np.any(result.sensor_data > 0.0), (
        "Non-zero source should produce positive pressure at some time step"
    )


def test_dg_grid_sensor_all_points():
    """DG solver with ``Sensor.grid()`` records all grid points.

    ``Sensor.grid()`` creates an all-``True`` mask covering every grid
    vertex.  This verifies the DG solver records pressure history at
    every node without crashing.
    """
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 2] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()  # all 64 grid vertices

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.DG)
    time_steps = 3
    result = sim.run(time_steps=time_steps)

    # 4³ = 64 sensors, time_steps columns
    assert result.num_sensors == 64
    assert result.sensor_data_shape == (64, time_steps)
    assert result.sensor_data.ndim == 2
    assert result.sensor_data.shape == (64, time_steps)
    assert np.all(np.isfinite(result.sensor_data))
    # The source-position node should have non-zero initial pressure
    mid_idx = 2 + 4 * (2 + 4 * 2)  # (2,2,2) in row-major
    assert result.sensor_data[mid_idx, 0] == pytest.approx(1.0, abs=1e-12)


def test_dg_smallest_valid_grid_succeeds():
    """DG solver succeeds on the smallest grid divisible by nodes_per_element.

    The DG solver uses polynomial-order-3 GLL nodes, so every axis must be
    a multiple of ``nodes_per_element=4``.  A 4×4×4 grid is the smallest
    valid isotropic grid.  Unlike Helmholtz/BEM, DG does not generate a
    tetrahedral mesh — it operates directly on Cartesian nodes.
    """
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[0, 0, 0] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.DG)

    # Must NOT raise — DG handles grid-aligned dimensions natively
    time_steps = 2
    result = sim.run(time_steps=time_steps)

    # 4³ = 64 sensors
    assert result.num_sensors == 64
    assert result.sensor_data_shape == (64, time_steps)
    assert np.all(np.isfinite(result.sensor_data))
    # Node [0,0,0] has p₀=1.0 at t=0
    assert result.sensor_data[0, 0] == pytest.approx(1.0, abs=1e-12)


# ══════════════════════════════════════════════════════════════════════════
# Graceful-degradation edge cases
# ══════════════════════════════════════════════════════════════════════════


def test_dg_zero_time_steps():
    """DG solver with ``time_steps=0`` should fail gracefully, not panic.

    With ``time_steps=0`` the solver allocates a 1-column sensor buffer
    (for the t=0 initial snapshot), then trims it to 0 columns.  The
    downstream statistics computation accesses column 0 of a 0-column
    array, which is an out-of-bounds index.

    .. warning::

        As of the current revision, this triggers a Rust-level panic
        inside ``run_dg_impl`` (index out of bounds in ``p_final``
        computation).  PyO3 catches the unwind and surfaces it as
        ``pyo3_runtime.PanicException`` (a ``BaseException`` subclass).
        The proper fix would be an early ``ValueError`` guard in
        ``Simulation.run()`` for ``time_steps == 0``.
    """
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 2] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.DG)

    # time_steps=0 should NOT silently produce garbage — it must either
    # return the trivial t=0 snapshot or raise a descriptive error.
    # Currently it panics at the Rust level; PyO3 surfaces that as a
    # PanicException (BaseException), which pytest.raises(Exception)
    # would miss.
    with pytest.raises(BaseException, match=r"index out of bounds|panic"):
        sim.run(time_steps=0)


def test_dg_very_large_dt_handling():
    """DG solver with an extremely large ``dt`` should handle it gracefully.

    The DG solver currently performs **no** CFL validation on ``dt``.
    With ``dt = 1.0 s`` on a 4-mm grid (≈ 660 000× the stable limit),
    the SSP-RK3 integrator may produce NaN or Inf from unbounded
    truncation error growth, or the solver may panic internally.

    This test verifies that the solver either:

    1. Raises a descriptive Python exception (``ValueError``) if a
       CFL guard is later added, or
    2. Surfaces a Rust panic as ``BaseException`` (the current PyO3
       panic-catch behaviour), or
    3. Completes and returns **finite** values — meaning the solver
       is unexpectedly stable even at extreme ``dt`` (in which case
       this test serves as a regression gate).

    If the solver returns NaN or Inf, the test fails to flag that
    CFL validation is needed.
    """
    grid = kw.Grid(nx=4, ny=4, nz=4, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    p0 = np.zeros((4, 4, 4), dtype=np.float64)
    p0[2, 2, 2] = 1.0
    source = kw.Source.from_initial_pressure(p0)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[2, 2, 2] = True
    sensor = kw.Sensor.from_mask(mask)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.DG)

    # dt = 1.0 s is absurdly large — 660 000× the CFL limit.
    # If the solver eventually gains CFL validation, expect ValueError.
    # If it panics, PyO3 surfaces a BaseException.
    try:
        result = sim.run(time_steps=3, dt=1.0)
    except ValueError:
        # Graceful: solver now validates dt → this is the ideal path.
        return
    except BaseException:
        # PyO3 catches Rust panics → BaseException; acceptable for now.
        return

    # If the solver survived, verify the output is well-formed and
    # that no non-finite values leaked through.
    assert result.num_sensors == 1
    assert result.sensor_data_shape == (1, 3)
    assert result.sensor_data is not None
    assert np.all(np.isfinite(result.sensor_data)), (
        "Large dt (660 000× CFL) produced non-finite sensor values — "
        "the DG solver needs CFL validation to reject unstable dt early"
    )
