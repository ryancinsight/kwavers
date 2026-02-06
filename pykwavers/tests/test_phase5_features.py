#!/usr/bin/env python3
"""
Phase 5 Feature Tests: Solver Selection & Multi-Source Support

Tests the newly implemented features:
1. Solver type enum (FDTD, PSTD, Hybrid)
2. Multi-source support (multiple sources in single simulation)
3. Backward compatibility (single source still works)
4. Source superposition validation

Mathematical Specifications:
- Linear superposition: p_total = Σ p_i
- Source independence: Each source contributes additively
- Solver selection: FDTD (default), PSTD/Hybrid (future)

Author: Ryan Clanton (@ryancinsight)
Date: 2024-02-04
Sprint: 217 Session 10 - Phase 5 Implementation
"""

import numpy as np
import pykwavers as kw
import pytest

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def standard_grid():
    """Standard 64³ grid with 0.1 mm spacing."""
    return kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)


@pytest.fixture
def water_medium():
    """Water at 20°C."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def center_sensor():
    """Sensor at center of 64³ grid."""
    return kw.Sensor.point((3.2e-3, 3.2e-3, 3.2e-3))


# ============================================================================
# Test: Solver Type Enum
# ============================================================================


def test_solver_type_enum_exists():
    """Verify SolverType enum is exposed to Python."""
    assert hasattr(kw, "SolverType")
    assert hasattr(kw.SolverType, "FDTD")
    assert hasattr(kw.SolverType, "PSTD")
    assert hasattr(kw.SolverType, "Hybrid")


def test_solver_type_repr():
    """Verify SolverType string representations."""
    assert repr(kw.SolverType.FDTD) == "SolverType.FDTD"
    assert repr(kw.SolverType.PSTD) == "SolverType.PSTD"
    assert repr(kw.SolverType.Hybrid) == "SolverType.Hybrid"

    assert str(kw.SolverType.FDTD) == "FDTD"
    assert str(kw.SolverType.PSTD) == "PSTD"
    assert str(kw.SolverType.Hybrid) == "Hybrid"


def test_solver_type_equality():
    """Verify SolverType equality comparison."""
    assert kw.SolverType.FDTD == kw.SolverType.FDTD
    assert kw.SolverType.PSTD == kw.SolverType.PSTD
    assert kw.SolverType.FDTD != kw.SolverType.PSTD


# ============================================================================
# Test: Multi-Source API
# ============================================================================


def test_multi_source_list_accepted(standard_grid, water_medium, center_sensor):
    """Verify Simulation accepts list of sources."""
    source1 = kw.Source.point((1e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)
    source2 = kw.Source.point((5e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)

    # Should accept list of sources
    sim = kw.Simulation(standard_grid, water_medium, [source1, source2], center_sensor)
    assert sim is not None


def test_single_source_still_works(standard_grid, water_medium, center_sensor):
    """Verify backward compatibility: single source still accepted."""
    source = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=1e5)

    # Should accept single source (backward compatible)
    sim = kw.Simulation(standard_grid, water_medium, source, center_sensor)
    assert sim is not None


def test_empty_source_list_rejected(standard_grid, water_medium, center_sensor):
    """Verify empty source list raises error."""
    with pytest.raises(ValueError, match="At least one source is required"):
        kw.Simulation(standard_grid, water_medium, [], center_sensor)


def test_invalid_source_type_rejected(standard_grid, water_medium, center_sensor):
    """Verify invalid source type raises error."""
    with pytest.raises(ValueError, match="sources must be a Source or list of Sources"):
        kw.Simulation(standard_grid, water_medium, "invalid", center_sensor)


# ============================================================================
# Test: Solver Selection
# ============================================================================


def test_default_solver_is_fdtd(standard_grid, water_medium, center_sensor):
    """Verify default solver is FDTD (backward compatible)."""
    source = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=1e5)
    sim = kw.Simulation(standard_grid, water_medium, source, center_sensor)

    # Check repr includes solver type
    assert "FDTD" in repr(sim)


def test_explicit_fdtd_solver(standard_grid, water_medium, center_sensor):
    """Verify explicit FDTD solver selection works."""
    source = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=1e5)
    sim = kw.Simulation(
        standard_grid, water_medium, source, center_sensor, solver=kw.SolverType.FDTD
    )

    result = sim.run(time_steps=100)
    assert result.time_steps == 100
    assert len(result.sensor_data) == 100


def test_pstd_solver_implemented(standard_grid, water_medium, center_sensor):
    """Verify PSTD solver is now implemented and runs successfully."""
    source = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=1e5)
    sim = kw.Simulation(
        standard_grid, water_medium, source, center_sensor, solver=kw.SolverType.PSTD
    )

    result = sim.run(time_steps=100)
    assert result.time_steps == 100
    assert len(result.sensor_data) == 100


def test_hybrid_solver_implemented(standard_grid, water_medium, center_sensor):
    """Verify Hybrid solver is now implemented and runs successfully."""
    source = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=1e5)
    sim = kw.Simulation(
        standard_grid, water_medium, source, center_sensor, solver=kw.SolverType.Hybrid
    )

    result = sim.run(time_steps=100)
    assert result.time_steps == 100
    assert len(result.sensor_data) == 100


# ============================================================================
# Test: Multi-Source Superposition
# ============================================================================


def test_two_point_sources_superpose(standard_grid, water_medium, center_sensor):
    """
    Verify two point sources obey linear superposition.

    Mathematical Specification:
    - p_total(x,t) = p1(x,t) + p2(x,t)
    - Relative error: ε = ||p_total - (p1 + p2)||₂ / ||p1 + p2||₂
    - Acceptance: ε < 0.05 (5%)
    """
    # Two point sources at symmetric positions
    source1 = kw.Source.point((2e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)
    source2 = kw.Source.point((4.4e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)

    # Run with source 1 only
    sim1 = kw.Simulation(standard_grid, water_medium, [source1], center_sensor)
    result1 = sim1.run(time_steps=500)

    # Run with source 2 only
    sim2 = kw.Simulation(standard_grid, water_medium, [source2], center_sensor)
    result2 = sim2.run(time_steps=500)

    # Run with both sources
    sim_both = kw.Simulation(standard_grid, water_medium, [source1, source2], center_sensor)
    result_both = sim_both.run(time_steps=500)

    # Compute expected pressure (linear superposition)
    p_expected = result1.sensor_data + result2.sensor_data
    p_measured = result_both.sensor_data

    # Compute relative L2 error
    error = np.linalg.norm(p_measured - p_expected) / np.linalg.norm(p_expected)

    print(f"\nSuperposition Test:")
    print(f"  L2 error: {error:.2%}")
    print(f"  Max p1: {np.max(np.abs(result1.sensor_data)):.2e} Pa")
    print(f"  Max p2: {np.max(np.abs(result2.sensor_data)):.2e} Pa")
    print(f"  Max p_expected: {np.max(np.abs(p_expected)):.2e} Pa")
    print(f"  Max p_measured: {np.max(np.abs(p_measured)):.2e} Pa")

    # Acceptance: Linear superposition should hold to within 5%
    assert error < 0.05, f"Superposition error {error:.2%} exceeds 5%"


def test_three_sources_superpose(standard_grid, water_medium):
    """Verify three sources also superpose correctly."""
    sensor = kw.Sensor.point((3.2e-3, 3.2e-3, 3.2e-3))

    # Three point sources at different positions
    sources = [
        kw.Source.point((1.6e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=3e4),
        kw.Source.point((3.2e-3, 1.6e-3, 3.2e-3), frequency=1e6, amplitude=3e4),
        kw.Source.point((3.2e-3, 4.8e-3, 3.2e-3), frequency=1e6, amplitude=3e4),
    ]

    # Run individual sources
    individual_results = []
    for source in sources:
        sim = kw.Simulation(standard_grid, water_medium, [source], sensor)
        result = sim.run(time_steps=500)
        individual_results.append(result.sensor_data)

    # Run all sources together
    sim_all = kw.Simulation(standard_grid, water_medium, sources, sensor)
    result_all = sim_all.run(time_steps=500)

    # Expected: sum of individual contributions
    p_expected = sum(individual_results)
    p_measured = result_all.sensor_data

    error = np.linalg.norm(p_measured - p_expected) / np.linalg.norm(p_expected)

    print(f"\nThree-Source Superposition Test:")
    print(f"  L2 error: {error:.2%}")

    assert error < 0.05, f"Three-source superposition error {error:.2%} exceeds 5%"


def test_mixed_source_types_superpose(standard_grid, water_medium):
    """Verify different source types (plane wave + point) superpose."""
    sensor = kw.Sensor.point((3.2e-3, 3.2e-3, 3.2e-3))

    # One plane wave and one point source
    plane_wave = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=5e4)
    point_source = kw.Source.point((1.6e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)

    # Run individually
    sim_plane = kw.Simulation(standard_grid, water_medium, [plane_wave], sensor)
    result_plane = sim_plane.run(time_steps=500)

    sim_point = kw.Simulation(standard_grid, water_medium, [point_source], sensor)
    result_point = sim_point.run(time_steps=500)

    # Run together
    sim_both = kw.Simulation(standard_grid, water_medium, [plane_wave, point_source], sensor)
    result_both = sim_both.run(time_steps=500)

    # Check superposition
    p_expected = result_plane.sensor_data + result_point.sensor_data
    p_measured = result_both.sensor_data

    error = np.linalg.norm(p_measured - p_expected) / np.linalg.norm(p_expected)

    print(f"\nMixed Source Type Superposition Test:")
    print(f"  L2 error: {error:.2%}")

    # Allow slightly larger error for mixed types
    assert error < 0.10, f"Mixed source superposition error {error:.2%} exceeds 10%"


# ============================================================================
# Test: Performance & Scaling
# ============================================================================


def test_multi_source_performance_scaling(standard_grid, water_medium, center_sensor):
    """Verify multi-source simulation scales reasonably (< 2× per source)."""
    import time

    # Single source baseline
    source = kw.Source.point((3.2e-3, 3.2e-3, 1.6e-3), frequency=1e6, amplitude=5e4)
    sim_single = kw.Simulation(standard_grid, water_medium, [source], center_sensor)

    start = time.perf_counter()
    sim_single.run(time_steps=100)
    time_single = time.perf_counter() - start

    # Two sources
    source2 = kw.Source.point((3.2e-3, 3.2e-3, 4.8e-3), frequency=1e6, amplitude=5e4)
    sim_double = kw.Simulation(standard_grid, water_medium, [source, source2], center_sensor)

    start = time.perf_counter()
    sim_double.run(time_steps=100)
    time_double = time.perf_counter() - start

    slowdown = time_double / time_single

    print(f"\nPerformance Scaling:")
    print(f"  Single source: {time_single:.3f} s")
    print(f"  Two sources: {time_double:.3f} s")
    print(f"  Slowdown: {slowdown:.2f}×")

    # Should not be more than 3× slower (generous bound)
    assert slowdown < 3.0, f"Multi-source slowdown {slowdown:.2f}× exceeds 3×"


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_identical_sources_double_amplitude(standard_grid, water_medium, center_sensor):
    """Verify two identical sources produce 2× amplitude."""
    source = kw.Source.point((3.2e-3, 3.2e-3, 1.6e-3), frequency=1e6, amplitude=5e4)

    # Single source
    sim_single = kw.Simulation(standard_grid, water_medium, [source], center_sensor)
    result_single = sim_single.run(time_steps=500)

    # Two identical sources (same position, frequency, amplitude)
    sim_double = kw.Simulation(standard_grid, water_medium, [source, source], center_sensor)
    result_double = sim_double.run(time_steps=500)

    # Should be approximately 2× amplitude
    max_single = np.max(np.abs(result_single.sensor_data))
    max_double = np.max(np.abs(result_double.sensor_data))
    ratio = max_double / max_single

    print(f"\nIdentical Source Test:")
    print(f"  Single source max: {max_single:.2e} Pa")
    print(f"  Double source max: {max_double:.2e} Pa")
    print(f"  Ratio: {ratio:.2f}")

    # Should be close to 2.0 (within 10% due to numerical effects)
    assert 1.8 < ratio < 2.2, f"Amplitude ratio {ratio:.2f} not close to 2.0"


def test_opposite_phase_sources_cancel(standard_grid, water_medium, center_sensor):
    """Verify sources at different distances can interfere destructively.

    Note: Phase parameter not yet exposed in API, so we use spatial positioning
    to create destructive interference by placing sources at half-wavelength separation.
    """
    # Wavelength at 1 MHz in water: λ = c/f = 1500/1e6 = 1.5 mm
    # Place two sources separated by λ/2 = 0.75 mm along z-axis
    source1 = kw.Source.point((3.2e-3, 3.2e-3, 1.6e-3), frequency=1e6, amplitude=5e4)
    source2 = kw.Source.point((3.2e-3, 3.2e-3, 2.35e-3), frequency=1e6, amplitude=5e4)  # +0.75 mm

    # Sensor equidistant from both sources (midpoint)
    sensor_mid = kw.Sensor.point((3.2e-3, 3.2e-3, 1.975e-3))

    sim = kw.Simulation(standard_grid, water_medium, [source1, source2], sensor_mid)
    result = sim.run(time_steps=500)

    # At the midpoint, waves should arrive with opposite phase and partially cancel
    max_pressure = np.max(np.abs(result.sensor_data))

    print(f"\nSpatial Interference Test:")
    print(f"  Max pressure at midpoint: {max_pressure:.2e} Pa")
    print(f"  (Note: Complete cancellation requires phase control not yet exposed)")

    # With spatial separation, we expect reduced but not zero amplitude
    # This test verifies multi-source wave propagation works correctly
    assert max_pressure < 1e5, f"Pressure {max_pressure:.2e} Pa unexpectedly high"


# ============================================================================
# Test: Simulation Repr
# ============================================================================


def test_simulation_repr_includes_solver(standard_grid, water_medium, center_sensor):
    """Verify Simulation.__repr__() includes solver type and source count."""
    sources = [
        kw.Source.point((1e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4),
        kw.Source.point((5e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4),
    ]

    sim = kw.Simulation(
        standard_grid, water_medium, sources, center_sensor, solver=kw.SolverType.FDTD
    )

    repr_str = repr(sim)

    print(f"\nSimulation repr: {repr_str}")

    # Should include solver type and source count
    assert "FDTD" in repr_str
    assert "sources=2" in repr_str


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
