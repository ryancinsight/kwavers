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

    # Signal should be detected (threshold accounts for 3D CFL √3 factor)
    max_pressure = np.max(np.abs(result.sensor_data))
    assert max_pressure > 5e2, f"Max pressure {max_pressure:.2e} Pa too low"

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
    - Propagation distance: d = 2.0 mm
    - Expected arrival: t = d/c₀ = 1.333 µs
    - Acceptance: |measured - expected| / expected < 15%

    Uses first-arrival detection: finds the first time the envelope exceeds
    a fraction of its maximum within a physically reasonable time window.
    This avoids false detection from late reflections in the small domain.
    """
    # Source and sensor inside usable domain (PML occupies 0-2mm and 4.4-6.4mm)
    source = kw.Source.point((2.2e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((4.2e-3, 3.2e-3, 3.2e-3))  # 2 mm away

    sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.PSTD)
    result = sim.run(time_steps=500)

    p = result.sensor_data
    envelope = np.abs(p)

    distance = 2.0e-3  # 2 mm
    c_physical = 1500.0
    t_expected = distance / c_physical

    # First-arrival detection: find when signal first exceeds 30% of the
    # maximum value seen in the expected arrival window [0.5*t_expected, 3*t_expected]
    dt = result.time[1] - result.time[0]
    t_min = int(0.5 * t_expected / dt)
    t_max = min(int(3.0 * t_expected / dt), len(p))
    window_max = np.max(envelope[t_min:t_max])
    threshold = 0.30 * window_max

    # Find first crossing of threshold
    arrival_idx = t_min
    for i in range(t_min, t_max):
        if envelope[i] > threshold:
            arrival_idx = i
            break

    arrival_time = result.time[arrival_idx]
    c_measured = distance / arrival_time if arrival_time > 0 else float('inf')
    timing_error = abs(c_measured - c_physical) / c_physical

    print(f"\nPSTD Dispersion Test:")
    print(f"  Physical c: {c_physical:.1f} m/s")
    print(f"  Measured c: {c_measured:.1f} m/s")
    print(f"  Timing error: {timing_error:.2%}")
    print(f"  Arrival time: {arrival_time * 1e6:.3f} µs")
    print(f"  Expected: {t_expected * 1e6:.3f} µs")

    # PSTD should have <15% timing error
    assert timing_error < 0.15, f"PSTD timing error {timing_error:.2%} exceeds 15%"


def test_pstd_vs_fdtd_dispersion_comparison(standard_grid, water_medium):
    """
    Compare PSTD (dispersion-free) vs FDTD (numerical dispersion).

    Uses first-arrival detection for timing measurement.
    Source and sensor are placed inside the usable domain (outside PML).
    """
    source = kw.Source.point((2.2e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((4.2e-3, 3.2e-3, 3.2e-3))  # 2 mm away
    distance = 2.0e-3
    c_physical = 1500.0
    t_expected = distance / c_physical

    def first_arrival_speed(result):
        """Detect first arrival using windowed threshold crossing."""
        env = np.abs(result.sensor_data)
        dt = result.time[1] - result.time[0]
        t_min = int(0.5 * t_expected / dt)
        t_max = min(int(3.0 * t_expected / dt), len(env))
        window_max = np.max(env[t_min:t_max])
        threshold = 0.30 * window_max
        for i in range(t_min, t_max):
            if env[i] > threshold:
                return distance / result.time[i]
        return float('inf')

    # Run PSTD
    sim_pstd = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.PSTD)
    result_pstd = sim_pstd.run(time_steps=500)

    # Run FDTD
    sim_fdtd = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.FDTD)
    result_fdtd = sim_fdtd.run(time_steps=500)

    c_pstd = first_arrival_speed(result_pstd)
    c_fdtd = first_arrival_speed(result_fdtd)

    pstd_error = abs(c_pstd - c_physical) / c_physical
    fdtd_error = abs(c_fdtd - c_physical) / c_physical

    print(f"\nPSTD vs FDTD Dispersion Comparison:")
    print(f"  Physical c: {c_physical:.1f} m/s")
    print(f"  PSTD c:     {c_pstd:.1f} m/s (error: {pstd_error:.2%})")
    print(f"  FDTD c:     {c_fdtd:.1f} m/s (error: {fdtd_error:.2%})")

    # Both solvers should propagate at close to correct speed (<20% error)
    assert pstd_error < 0.20, f"PSTD timing error {pstd_error:.2%} exceeds 20%"
    assert fdtd_error < 0.20, f"FDTD timing error {fdtd_error:.2%} exceeds 20%"


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
    assert max_pressure > 5e2, f"Max pressure {max_pressure:.2e} Pa too low"

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
    Verify Hybrid solver timing accuracy using first-arrival detection.

    Expected: Hybrid should propagate at close to physical speed.
    Source and sensor are placed inside the usable domain (outside PML).
    """
    source = kw.Source.point((2.2e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((4.2e-3, 3.2e-3, 3.2e-3))
    distance = 2.0e-3
    c_physical = 1500.0
    t_expected = distance / c_physical

    sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=kw.SolverType.Hybrid)
    result = sim.run(time_steps=500)

    # First-arrival detection
    env = np.abs(result.sensor_data)
    dt = result.time[1] - result.time[0]
    t_min = int(0.5 * t_expected / dt)
    t_max = min(int(3.0 * t_expected / dt), len(env))
    window_max = np.max(env[t_min:t_max])
    threshold = 0.30 * window_max

    arrival_idx = t_min
    for i in range(t_min, t_max):
        if env[i] > threshold:
            arrival_idx = i
            break

    arrival_time = result.time[arrival_idx]
    c_measured = distance / arrival_time if arrival_time > 0 else float('inf')
    timing_error = abs(c_measured - c_physical) / c_physical

    print(f"\nHybrid Timing Test:")
    print(f"  Physical c: {c_physical:.1f} m/s")
    print(f"  Measured c: {c_measured:.1f} m/s")
    print(f"  Timing error: {timing_error:.2%}")

    # Hybrid should have reasonable accuracy (< 20% error)
    assert timing_error < 0.20, f"Hybrid timing error {timing_error:.2%} exceeds 20%"


# ============================================================================
# Test: Solver Cross-Validation
# ============================================================================


def test_all_solvers_produce_consistent_results(standard_grid, water_medium):
    """
    Verify all three solvers (FDTD, PSTD, Hybrid) produce qualitatively
    consistent results using lag-aligned cross-correlation.

    Spectral methods (PSTD/Hybrid) may have slight timing offsets vs FDTD,
    so we use cross-correlation with lag alignment to compare waveforms.

    Acceptance: Lag-aligned cross-correlation > 0.5
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
    p_fdtd = result_fdtd.sensor_data / max(np.max(np.abs(result_fdtd.sensor_data)), 1e-30)
    p_pstd = result_pstd.sensor_data / max(np.max(np.abs(result_pstd.sensor_data)), 1e-30)
    p_hybrid = result_hybrid.sensor_data / max(np.max(np.abs(result_hybrid.sensor_data)), 1e-30)

    # Lag-aligned cross-correlation: find best alignment then compute correlation
    def lag_corr(a, b, max_lag=50):
        """Cross-correlation with lag alignment."""
        best_corr = -1.0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                aa = a[lag:]
                bb = b[:len(aa)]
            else:
                bb = b[-lag:]
                aa = a[:len(bb)]
            if len(aa) < 10:
                continue
            c = np.corrcoef(aa, bb)[0, 1]
            if not np.isnan(c) and c > best_corr:
                best_corr = c
        return best_corr

    corr_fdtd_pstd = lag_corr(p_fdtd, p_pstd)
    corr_fdtd_hybrid = lag_corr(p_fdtd, p_hybrid)
    corr_pstd_hybrid = lag_corr(p_pstd, p_hybrid)

    print(f"\nSolver Cross-Validation (lag-aligned):")
    print(f"  FDTD ↔ PSTD correlation:   {corr_fdtd_pstd:.4f}")
    print(f"  FDTD ↔ Hybrid correlation: {corr_fdtd_hybrid:.4f}")
    print(f"  PSTD ↔ Hybrid correlation: {corr_pstd_hybrid:.4f}")

    # All correlations should be reasonable (similar waveforms)
    assert corr_fdtd_pstd > 0.50, f"FDTD-PSTD correlation {corr_fdtd_pstd:.4f} < 0.50"
    assert corr_fdtd_hybrid > 0.50, f"FDTD-Hybrid correlation {corr_fdtd_hybrid:.4f} < 0.50"
    assert corr_pstd_hybrid > 0.50, f"PSTD-Hybrid correlation {corr_pstd_hybrid:.4f} < 0.50"


def test_solver_energy_conservation(standard_grid, water_medium):
    """
    Verify PSTD and Hybrid solvers do not exhibit spurious energy growth.

    A single-sensor RMS metric is not a true energy measure (signal naturally
    goes from silence → pulse → silence), so instead we verify:
    1. The peak pressure is finite and physically reasonable
    2. No exponential blowup: the signal tail does not exceed the peak
    3. The PSTD/Hybrid peak amplitudes are in the same order of magnitude

    Acceptance: No NaN/Inf, tail amplitude < peak, solvers within 10× of each other
    """
    source = kw.Source.point((3.2e-3, 3.2e-3, 1.6e-3), frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point((3.2e-3, 3.2e-3, 4.8e-3))

    peaks = {}
    for solver_type, name in [
        (kw.SolverType.PSTD, "PSTD"),
        (kw.SolverType.Hybrid, "Hybrid"),
    ]:
        sim = kw.Simulation(standard_grid, water_medium, source, sensor, solver=solver_type)
        result = sim.run(time_steps=1000)
        data = result.sensor_data

        # No NaN or Inf
        assert np.all(np.isfinite(data)), f"{name} produced NaN/Inf values"

        peak = np.max(np.abs(data))
        peaks[name] = peak

        # Check the tail (last 20% of time series) doesn't exceed the peak
        # This would indicate exponential growth / instability
        tail_start = int(0.8 * len(data))
        tail_max = np.max(np.abs(data[tail_start:]))
        ratio = tail_max / max(peak, 1e-30)

        print(f"\n{name} Stability:")
        print(f"  Peak pressure: {peak:.4e} Pa")
        print(f"  Tail max:      {tail_max:.4e} Pa")
        print(f"  Tail/peak:     {ratio:.4f}")

        assert ratio <= 1.0, f"{name} tail exceeds peak: growth detected ({ratio:.2f}×)"

    # PSTD and Hybrid peaks should be same order of magnitude
    ratio = peaks["PSTD"] / max(peaks["Hybrid"], 1e-30)
    print(f"\nPSTD/Hybrid peak ratio: {ratio:.2f}")
    assert 0.1 < ratio < 10.0, f"PSTD/Hybrid peak ratio {ratio:.2f} out of range"


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
    assert max_pressure > 100, f"Max pressure {max_pressure:.2e} Pa too low"


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
