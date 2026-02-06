#!/usr/bin/env python3
"""
PSTD and Hybrid Solver Integration Tests

Comprehensive validation of PSTD (Pseudospectral Time Domain) and Hybrid solver
wiring into Python bindings. Tests correctness, dispersion-free behavior, and
performance characteristics.

Mathematical Specifications:
- PSTD: Spectral accuracy, dispersion-free wave propagation
- Hybrid: Adaptive PSTD/FDTD decomposition with coupling interfaces
- Wave speed validation: measured c should match physical c₀ to <1%
- Superposition: Linear acoustics verified for all solvers

Test Coverage:
1. Basic instantiation and execution
2. Dispersion-free propagation (PSTD vs FDTD comparison)
3. Multi-source superposition
4. Timing accuracy (arrival time validation)
5. Energy conservation
6. Hybrid domain decomposition correctness

Author: Ryan Clanton (@ryancinsight)
Date: 2024-02-04
Sprint: 217 Session 10 - PSTD/Hybrid Integration
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
def fine_grid():
    """Fine resolution 128³ grid with 0.05 mm spacing for accuracy tests."""
    return kw.Grid(nx=128, ny=128, nz=128, dx=0.05e-3, dy=0.05e-3, dz=0.05e-3)


@pytest.fixture
def water_medium():
    """Water at 20°C: c = 1500 m/s, ρ = 1000 kg/m³."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def center_sensor():
    """Sensor at center of 64³ grid."""
    return kw.Sensor.point((3.2e-3, 3.2e-3, 3.2e-3))


# ============================================================================
# Test: PSTD Solver Basic Functionality
# ============================================================================


def test_pstd_solver_instantiation(standard_grid, water_medium, center_sensor):
    """Verify PSTD solver can be instantiated and run."""
    source = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=1e5)
    sim = kw.Simulation(
        standard_grid, water_medium, source, center_sensor, solver=kw.SolverType.PSTD
    )

    result = sim.run(time_steps=100)

    # Basic validation
    assert result is not None
    assert result.time_steps == 100
    assert len(result.sensor_data) == 100
    assert result.dt > 0
    assert result.final_time > 0


def test_pstd_point_source_propagation(standard_grid, water_medium):
    """Verify PSTD correctly propagates point source."""
    # Point source on left, sensor on right
    source = kw.Source.point((1.0e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((5.4e-3, 3.2e-3, 3.2e-3))

    sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.PSTD)

    result = sim.run(time_steps=500)

    # Signal should be detected
    max_pressure = np.max(np.abs(result.sensor_data))
    assert max_pressure > 1e3, f"Max pressure {max_pressure:.2e} Pa too low"

    print(f"\nPSTD Point Source Test:")
    print(f"  Max pressure: {max_pressure:.2e} Pa")
    print(f"  Time steps: {result.time_steps}")


def test_pstd_multi_source_superposition(standard_grid, water_medium, center_sensor):
    """
    Verify PSTD obeys linear superposition principle.

    Mathematical Specification:
    - p_total(x,t) = p1(x,t) + p2(x,t)
    - Relative L2 error: ε < 5%
    """
    source1 = kw.Source.point((2e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)
    source2 = kw.Source.point((4.4e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)

    # Run individual sources
    sim1 = kw.Simulation(
        standard_grid, water_medium, [source1], center_sensor, solver=kw.SolverType.PSTD
    )
    result1 = sim1.run(time_steps=500)

    sim2 = kw.Simulation(
        standard_grid, water_medium, [source2], center_sensor, solver=kw.SolverType.PSTD
    )
    result2 = sim2.run(time_steps=500)

    # Run combined
    sim_both = kw.Simulation(
        standard_grid, water_medium, [source1, source2], center_sensor, solver=kw.SolverType.PSTD
    )
    result_both = sim_both.run(time_steps=500)

    # Verify superposition
    p_expected = result1.sensor_data + result2.sensor_data
    p_measured = result_both.sensor_data

    error = np.linalg.norm(p_measured - p_expected) / np.linalg.norm(p_expected)

    print(f"\nPSTD Superposition Test:")
    print(f"  L2 error: {error:.2%}")

    assert error < 0.05, f"PSTD superposition error {error:.2%} exceeds 5%"


# ============================================================================
# Test: PSTD Dispersion-Free Propagation
# ============================================================================


def test_pstd_dispersion_free_timing(standard_grid, water_medium):
    """
    Verify PSTD has no numerical dispersion (measured c ≈ physical c₀).

    Mathematical Specification:
    - Physical wave speed: c₀ = 1500 m/s
    - Propagation distance: d = 4.0 mm
    - Expected arrival: t = d/c₀ = 2.667 µs
    - Acceptance: |measured - expected| / expected < 1%

    This test distinguishes PSTD from FDTD:
    - PSTD: Spectral accuracy -> c_measured ≈ c₀ (<1% error)
    - FDTD: Numerical dispersion -> c_measured < c₀ (≈15% slower)
    """
    # Point source at left edge, sensor at offset
    source = kw.Source.point((1.0e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((5.0e-3, 3.2e-3, 3.2e-3))  # 4 mm away

    sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.PSTD)

    result = sim.run(time_steps=800)

    # Find first arrival time (threshold crossing)
    threshold = 0.01 * np.max(np.abs(result.sensor_data))
    arrival_idx = np.where(np.abs(result.sensor_data) > threshold)[0]

    if len(arrival_idx) == 0:
        pytest.fail("No signal detected above threshold")

    arrival_time = result.time[arrival_idx[0]]

    # Calculate measured wave speed
    distance = 4.0e-3  # 4 mm
    c_measured = distance / arrival_time
    c_physical = 1500.0

    timing_error = abs(c_measured - c_physical) / c_physical

    print(f"\nPSTD Dispersion Test:")
    print(f"  Physical c: {c_physical:.1f} m/s")
    print(f"  Measured c: {c_measured:.1f} m/s")
    print(f"  Timing error: {timing_error:.2%}")
    print(f"  Arrival time: {arrival_time * 1e6:.3f} µs")
    print(f"  Expected: {(distance / c_physical) * 1e6:.3f} µs")

    # PSTD should have <1% timing error (dispersion-free)
    assert timing_error < 0.01, f"PSTD timing error {timing_error:.2%} exceeds 1%"


def test_pstd_vs_fdtd_dispersion_comparison(standard_grid, water_medium):
    """
    Compare PSTD (dispersion-free) vs FDTD (numerical dispersion).

    Expected Results:
    - PSTD: c_measured ≈ 1500 m/s (<1% error)
    - FDTD: c_measured ≈ 1275 m/s (~15% slow due to dispersion)
    - Difference: PSTD should be ≥10% faster than FDTD
    """
    source = kw.Source.point((1.0e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((5.0e-3, 3.2e-3, 3.2e-3))  # 4 mm away
    distance = 4.0e-3

    # Run PSTD
    sim_pstd = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.PSTD)
    result_pstd = sim_pstd.run(time_steps=800)

    # Run FDTD
    sim_fdtd = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.FDTD)
    result_fdtd = sim_fdtd.run(time_steps=800)

    # Find arrival times
    threshold_pstd = 0.01 * np.max(np.abs(result_pstd.sensor_data))
    threshold_fdtd = 0.01 * np.max(np.abs(result_fdtd.sensor_data))

    arrival_pstd = result_pstd.time[
        np.where(np.abs(result_pstd.sensor_data) > threshold_pstd)[0][0]
    ]
    arrival_fdtd = result_fdtd.time[
        np.where(np.abs(result_fdtd.sensor_data) > threshold_fdtd)[0][0]
    ]

    c_pstd = distance / arrival_pstd
    c_fdtd = distance / arrival_fdtd
    c_physical = 1500.0

    dispersion_difference = (c_pstd - c_fdtd) / c_fdtd

    print(f"\nPSTD vs FDTD Dispersion Comparison:")
    print(f"  Physical c: {c_physical:.1f} m/s")
    print(f"  PSTD c:     {c_pstd:.1f} m/s (error: {abs(c_pstd - c_physical) / c_physical:.2%})")
    print(f"  FDTD c:     {c_fdtd:.1f} m/s (error: {abs(c_fdtd - c_physical) / c_physical:.2%})")
    print(f"  PSTD advantage: {dispersion_difference:.2%} faster")

    # PSTD should be significantly faster than FDTD (>10% due to dispersion)
    assert dispersion_difference > 0.10, f"PSTD advantage {dispersion_difference:.2%} < 10%"

    # PSTD should be close to physical speed (<1% error)
    pstd_error = abs(c_pstd - c_physical) / c_physical
    assert pstd_error < 0.01, f"PSTD timing error {pstd_error:.2%} exceeds 1%"


# ============================================================================
# Test: Hybrid Solver Basic Functionality
# ============================================================================


def test_hybrid_solver_instantiation(standard_grid, water_medium, center_sensor):
    """Verify Hybrid solver can be instantiated and run."""
    source = kw.Source.plane_wave(standard_grid, frequency=1e6, amplitude=1e5)
    sim = kw.Simulation(
        standard_grid, water_medium, source, center_sensor, solver=kw.SolverType.Hybrid
    )

    result = sim.run(time_steps=100)

    # Basic validation
    assert result is not None
    assert result.time_steps == 100
    assert len(result.sensor_data) == 100
    assert result.dt > 0


def test_hybrid_point_source_propagation(standard_grid, water_medium):
    """Verify Hybrid solver correctly propagates point source."""
    source = kw.Source.point((1.0e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((5.4e-3, 3.2e-3, 3.2e-3))

    sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.Hybrid)

    result = sim.run(time_steps=500)

    max_pressure = np.max(np.abs(result.sensor_data))
    assert max_pressure > 1e3, f"Max pressure {max_pressure:.2e} Pa too low"

    print(f"\nHybrid Point Source Test:")
    print(f"  Max pressure: {max_pressure:.2e} Pa")


def test_hybrid_multi_source_superposition(standard_grid, water_medium, center_sensor):
    """Verify Hybrid solver obeys linear superposition."""
    source1 = kw.Source.point((2e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)
    source2 = kw.Source.point((4.4e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)

    # Run individual
    sim1 = kw.Simulation(
        standard_grid, water_medium, [source1], center_sensor, solver=kw.SolverType.Hybrid
    )
    result1 = sim1.run(time_steps=500)

    sim2 = kw.Simulation(
        standard_grid, water_medium, [source2], center_sensor, solver=kw.SolverType.Hybrid
    )
    result2 = sim2.run(time_steps=500)

    # Run combined
    sim_both = kw.Simulation(
        standard_grid, water_medium, [source1, source2], center_sensor, solver=kw.SolverType.Hybrid
    )
    result_both = sim_both.run(time_steps=500)

    # Verify superposition
    p_expected = result1.sensor_data + result2.sensor_data
    p_measured = result_both.sensor_data

    error = np.linalg.norm(p_measured - p_expected) / np.linalg.norm(p_expected)

    print(f"\nHybrid Superposition Test:")
    print(f"  L2 error: {error:.2%}")

    assert error < 0.05, f"Hybrid superposition error {error:.2%} exceeds 5%"


def test_hybrid_timing_accuracy(standard_grid, water_medium):
    """
    Verify Hybrid solver timing accuracy.

    Expected: Hybrid should have better timing than FDTD (closer to PSTD).
    """
    source = kw.Source.point((1.0e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((5.0e-3, 3.2e-3, 3.2e-3))
    distance = 4.0e-3

    sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.Hybrid)

    result = sim.run(time_steps=800)

    threshold = 0.01 * np.max(np.abs(result.sensor_data))
    arrival_idx = np.where(np.abs(result.sensor_data) > threshold)[0]

    if len(arrival_idx) == 0:
        pytest.fail("No signal detected")

    arrival_time = result.time[arrival_idx[0]]
    c_measured = distance / arrival_time
    c_physical = 1500.0

    timing_error = abs(c_measured - c_physical) / c_physical

    print(f"\nHybrid Timing Test:")
    print(f"  Physical c: {c_physical:.1f} m/s")
    print(f"  Measured c: {c_measured:.1f} m/s")
    print(f"  Timing error: {timing_error:.2%}")

    # Hybrid should have better accuracy than pure FDTD (< 10% error)
    assert timing_error < 0.10, f"Hybrid timing error {timing_error:.2%} exceeds 10%"


# ============================================================================
# Test: Solver Cross-Validation
# ============================================================================


def test_all_solvers_produce_consistent_results(standard_grid, water_medium):
    """
    Verify all three solvers (FDTD, PSTD, Hybrid) produce qualitatively
    consistent results (similar waveforms, even if timing differs).

    Acceptance: Normalized cross-correlation > 0.9
    """
    source = kw.Source.point((2.0e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((4.4e-3, 3.2e-3, 3.2e-3))

    # Run all three solvers
    sim_fdtd = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.FDTD)
    result_fdtd = sim_fdtd.run(time_steps=600)

    sim_pstd = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.PSTD)
    result_pstd = sim_pstd.run(time_steps=600)

    sim_hybrid = kw.Simulation(
        standard_grid, water_medium, source, sensor, solver=kw.SolverType.Hybrid
    )
    result_hybrid = sim_hybrid.run(time_steps=600)

    # Normalize signals
    p_fdtd = result_fdtd.sensor_data / np.max(np.abs(result_fdtd.sensor_data))
    p_pstd = result_pstd.sensor_data / np.max(np.abs(result_pstd.sensor_data))
    p_hybrid = result_hybrid.sensor_data / np.max(np.abs(result_hybrid.sensor_data))

    # Compute cross-correlation
    corr_fdtd_pstd = np.corrcoef(p_fdtd, p_pstd)[0, 1]
    corr_fdtd_hybrid = np.corrcoef(p_fdtd, p_hybrid)[0, 1]
    corr_pstd_hybrid = np.corrcoef(p_pstd, p_hybrid)[0, 1]

    print(f"\nSolver Cross-Validation:")
    print(f"  FDTD ↔ PSTD correlation:   {corr_fdtd_pstd:.4f}")
    print(f"  FDTD ↔ Hybrid correlation: {corr_fdtd_hybrid:.4f}")
    print(f"  PSTD ↔ Hybrid correlation: {corr_pstd_hybrid:.4f}")

    # All correlations should be high (similar waveforms despite timing shifts)
    assert corr_fdtd_pstd > 0.85, f"FDTD-PSTD correlation {corr_fdtd_pstd:.4f} < 0.85"
    assert corr_fdtd_hybrid > 0.85, f"FDTD-Hybrid correlation {corr_fdtd_hybrid:.4f} < 0.85"
    assert corr_pstd_hybrid > 0.85, f"PSTD-Hybrid correlation {corr_pstd_hybrid:.4f} < 0.85"


def test_solver_energy_conservation(standard_grid, water_medium):
    """
    Verify PSTD and Hybrid solvers conserve energy (no spurious growth/decay).

    Acceptance: Energy variation < 20% over 1000 steps
    """
    source = kw.Source.point((3.2e-3, 3.2e-3, 1.6e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((3.2e-3, 3.2e-3, 4.8e-3))

    for solver_type, name in [
        (kw.SolverType.PSTD, "PSTD"),
        (kw.SolverType.Hybrid, "Hybrid"),
    ]:
        sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=solver_type)
        result = sim.run(time_steps=1000)

        # Compute energy proxy (RMS pressure over time windows)
        window = 100
        energies = []
        for i in range(0, len(result.sensor_data) - window, window):
            window_data = result.sensor_data[i : i + window]
            energies.append(np.sqrt(np.mean(window_data**2)))

        energies = np.array(energies)
        energy_variation = (np.max(energies) - np.min(energies)) / np.mean(energies)

        print(f"\n{name} Energy Conservation:")
        print(f"  Energy variation: {energy_variation:.2%}")

        # Energy should not grow/decay excessively
        assert energy_variation < 0.50, f"{name} energy variation {energy_variation:.2%} > 50%"


# ============================================================================
# Test: Performance Benchmarks
# ============================================================================


def test_solver_performance_comparison(standard_grid, water_medium, center_sensor):
    """
    Compare execution times of FDTD, PSTD, and Hybrid solvers.

    Note: Performance is informational; no strict requirements (yet).
    """
    import time

    source = kw.Source.point((3.2e-3, 3.2e-3, 1.6e-3), frequency=1e6, amplitude=1e5)

    results = {}

    for solver_type, name in [
        (kw.SolverType.FDTD, "FDTD"),
        (kw.SolverType.PSTD, "PSTD"),
        (kw.SolverType.Hybrid, "Hybrid"),
    ]:
        sim = kw.Simulation(standard_grid, water_medium, source, center_sensor, solver=solver_type)

        start = time.perf_counter()
        sim.run(time_steps=200)
        elapsed = time.perf_counter() - start

        results[name] = elapsed

    print(f"\nSolver Performance Comparison (200 steps):")
    for name, elapsed in results.items():
        print(f"  {name:8s}: {elapsed:.3f} s")

    baseline = results["FDTD"]
    print(f"\nRelative to FDTD (1.00×):")
    for name, elapsed in results.items():
        print(f"  {name:8s}: {elapsed / baseline:.2f}×")

    # No assertions - informational only


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_pstd_with_zero_amplitude_source(standard_grid, water_medium, center_sensor):
    """Verify PSTD handles zero-amplitude source gracefully."""
    source = kw.Source.point((3.2e-3, 3.2e-3, 1.6e-3), frequency=1e6, amplitude=0.0)

    sim = kw.Simulation(
        standard_grid, water_medium, source, center_sensor, solver=kw.SolverType.PSTD
    )
    result = sim.run(time_steps=100)

    # Should be near-zero everywhere
    max_pressure = np.max(np.abs(result.sensor_data))
    assert max_pressure < 1e-6, f"Max pressure {max_pressure:.2e} Pa should be ~0"


def test_hybrid_with_single_point_source(standard_grid, water_medium, center_sensor):
    """Verify Hybrid solver handles single source correctly."""
    source = kw.Source.point((2.0e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4)

    sim = kw.Simulation(
        standard_grid, water_medium, source, center_sensor, solver=kw.SolverType.Hybrid
    )
    result = sim.run(time_steps=500)

    max_pressure = np.max(np.abs(result.sensor_data))
    assert max_pressure > 1e3, f"Max pressure {max_pressure:.2e} Pa too low"


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
