"""
Quick PSTD Diagnostic Test

Fast diagnostic to validate PSTD source injection with small grid and short duration.
This test is designed to run in < 30 seconds to quickly verify behavior.

Expected Results:
- Wave propagates from z=0 plane
- Sensor records pressure within 20% of source amplitude
- PSTD and FDTD show similar behavior

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-05
Sprint: 217 - k-Wave Comparison & Validation
"""

import os
import sys
from pathlib import Path

import numpy as np

# Enable Rust tracing (info level for speed)
os.environ["RUST_LOG"] = "kwavers=info"

import pykwavers as kw

# Initialize tracing
# kw.init_tracing()


def run_quick_test(solver_type, grid_size, spacing, num_steps, dt):
    """
    Run quick diagnostic test.

    Parameters
    ----------
    solver_type : kw.SolverType
        FDTD or PSTD
    grid_size : tuple
        (nx, ny, nz)
    spacing : float
        Grid spacing [m]
    num_steps : int
        Number of time steps
    dt : float
        Time step [s]

    Returns
    -------
    dict
        Results with pressure data and metrics
    """
    nx, ny, nz = grid_size
    dx = dy = dz = spacing

    print(f"\n{'=' * 60}")
    print(f"Running {solver_type} - Quick Diagnostic")
    print(f"{'=' * 60}")
    print(f"Grid: {nx}x{ny}x{nz} ({nx * ny * nz:,} points)")
    print(f"Spacing: {dx * 1e3:.2f} mm")
    print(f"Steps: {num_steps}, dt: {dt * 1e9:.2f} ns")
    print(f"Duration: {num_steps * dt * 1e6:.2f} us")

    # Create grid
    grid = kw.Grid(nx, ny, nz, dx, dy, dz)

    # Medium (water)
    c0 = 1500.0
    medium = kw.Medium.homogeneous(sound_speed=c0, density=1000.0, absorption=0.0)

    # Plane wave source at z=0
    f0 = 1.0e6  # 1 MHz
    amp = 100e3  # 100 kPa
    source = kw.Source.plane_wave(grid, frequency=f0, amplitude=amp)

    # Sensor near center, but far enough from source
    # Place at z = 60% of domain to allow propagation
    sensor_x = nx // 2 * dx
    sensor_y = ny // 2 * dy
    sensor_z = int(nz * 0.6) * dz
    sensor_pos = (sensor_x, sensor_y, sensor_z)
    sensor = kw.Sensor.point(position=sensor_pos)

    print(f"Source: {f0 / 1e6:.1f} MHz plane wave, {amp / 1e3:.0f} kPa")
    print(
        f"Sensor: ({sensor_pos[0] * 1e3:.1f}, {sensor_pos[1] * 1e3:.1f}, {sensor_pos[2] * 1e3:.1f}) mm"
    )
    print(f"Distance from source: {sensor_z * 1e3:.1f} mm")

    # Expected arrival time
    t_arrival = sensor_z / c0
    print(f"Expected wave arrival: {t_arrival * 1e6:.2f} us")

    # Run simulation
    print("Running simulation...")
    sim = kw.Simulation(grid, medium, source, sensor, solver=solver_type)
    result = sim.run(time_steps=num_steps, dt=dt)

    print(f"OK Complete in {num_steps} steps")

    # Analyze sensor data
    sensor_data = result.sensor_data
    time = result.time

    p_max = np.max(np.abs(sensor_data))
    p_mean = np.mean(np.abs(sensor_data))

    # Find arrival time (first significant pressure)
    threshold = amp * 0.05  # 5% threshold
    arrival_idx = np.where(np.abs(sensor_data) > threshold)[0]
    if len(arrival_idx) > 0:
        t_arrival_sim = time[arrival_idx[0]]
    else:
        t_arrival_sim = np.nan

    print(f"\nResults:")
    print(f"  Max pressure: {p_max / 1e3:.1f} kPa")
    print(f"  Mean pressure: {p_mean / 1e3:.2f} kPa")
    print(
        f"  Simulated arrival: {t_arrival_sim * 1e6:.2f} us"
        if not np.isnan(t_arrival_sim)
        else "  No arrival detected"
    )
    print(
        f"  Arrival error: {(t_arrival_sim - t_arrival) / t_arrival * 100:+.1f}%"
        if not np.isnan(t_arrival_sim)
        else ""
    )
    print(f"  Amplitude error: {(p_max - amp) / amp * 100:+.1f}%")

    return {
        "solver": str(solver_type),
        "time": time,
        "sensor_data": sensor_data,
        "p_max": p_max,
        "p_mean": p_mean,
        "t_arrival_sim": t_arrival_sim,
        "t_arrival_theory": t_arrival,
        "sensor_pos": sensor_pos,
        "amplitude": amp,
    }


def compare_results(fdtd_results, pstd_results):
    """Compare FDTD and PSTD results."""
    print(f"\n{'=' * 60}")
    print("COMPARISON: FDTD vs PSTD")
    print(f"{'=' * 60}")

    # Amplitude comparison
    fdtd_amp = fdtd_results["p_max"]
    pstd_amp = pstd_results["p_max"]
    expected_amp = fdtd_results["amplitude"]

    fdtd_error = (fdtd_amp - expected_amp) / expected_amp * 100
    pstd_error = (pstd_amp - expected_amp) / expected_amp * 100
    relative_diff = (pstd_amp - fdtd_amp) / fdtd_amp * 100

    print(f"\nAmplitude:")
    print(f"  Expected:  {expected_amp / 1e3:.1f} kPa")
    print(f"  FDTD:      {fdtd_amp / 1e3:.1f} kPa (error: {fdtd_error:+.1f}%)")
    print(f"  PSTD:      {pstd_amp / 1e3:.1f} kPa (error: {pstd_error:+.1f}%)")
    print(f"  PSTD vs FDTD: {relative_diff:+.1f}%")

    # Arrival time comparison
    fdtd_arrival = fdtd_results["t_arrival_sim"]
    pstd_arrival = pstd_results["t_arrival_sim"]
    expected_arrival = fdtd_results["t_arrival_theory"]

    print(f"\nArrival Time:")
    print(f"  Expected:  {expected_arrival * 1e6:.2f} us")
    if not np.isnan(fdtd_arrival):
        print(
            f"  FDTD:      {fdtd_arrival * 1e6:.2f} us (error: {(fdtd_arrival - expected_arrival) / expected_arrival * 100:+.1f}%)"
        )
    else:
        print(f"  FDTD:      No arrival detected")
    if not np.isnan(pstd_arrival):
        print(
            f"  PSTD:      {pstd_arrival * 1e6:.2f} us (error: {(pstd_arrival - expected_arrival) / expected_arrival * 100:+.1f}%)"
        )
    else:
        print(f"  PSTD:      No arrival detected")

    # Validation criteria
    print(f"\nValidation (tolerance: +/-20%):")

    fdtd_amp_pass = abs(fdtd_error) < 20
    pstd_amp_pass = abs(pstd_error) < 20
    fdtd_arrival_pass = (
        abs((fdtd_arrival - expected_arrival) / expected_arrival) < 0.2
        if not np.isnan(fdtd_arrival)
        else False
    )
    pstd_arrival_pass = (
        abs((pstd_arrival - expected_arrival) / expected_arrival) < 0.2
        if not np.isnan(pstd_arrival)
        else False
    )

    print(f"  FDTD amplitude:    {'PASS' if fdtd_amp_pass else 'FAIL'}")
    print(f"  FDTD arrival:      {'PASS' if fdtd_arrival_pass else 'FAIL'}")
    print(f"  PSTD amplitude:    {'PASS' if pstd_amp_pass else 'FAIL'}")
    print(f"  PSTD arrival:      {'PASS' if pstd_arrival_pass else 'FAIL'}")

    overall_pass = fdtd_amp_pass and pstd_amp_pass and fdtd_arrival_pass and pstd_arrival_pass

    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

    return overall_pass


def main():
    """Run quick diagnostic test."""
    print("=" * 60)
    print("PSTD Quick Diagnostic Test")
    print("=" * 60)
    print("\nObjective: Fast validation of PSTD source injection")
    print("Expected runtime: < 30 seconds")

    # Small grid for speed
    nx = ny = nz = 64
    dx = 0.1e-3  # 0.1 mm

    # Time parameters
    c0 = 1500.0
    cfl = 0.3
    dt = cfl * dx / (c0 * np.sqrt(3))

    # Short duration - just enough for wave to propagate through domain
    # Domain size: 64 * 0.1mm = 6.4 mm
    # Propagation time: 6.4mm / 1500 m/s = 4.27 us
    # Add margin: 8 us total
    duration = 8e-6
    num_steps = int(duration / dt)

    print(f"\nConfiguration:")
    print(f"  Grid: {nx}x{ny}x{nz} = {nx**3:,} points")
    print(f"  Spacing: {dx * 1e3:.2f} mm")
    print(f"  Time step: {dt * 1e9:.2f} ns")
    print(f"  Steps: {num_steps}")
    print(f"  Duration: {duration * 1e6:.1f} us")

    # Run FDTD
    fdtd_results = run_quick_test(kw.SolverType.FDTD, (nx, ny, nz), dx, num_steps, dt)

    # Run PSTD
    pstd_results = run_quick_test(kw.SolverType.PSTD, (nx, ny, nz), dx, num_steps, dt)

    # Compare
    overall_pass = compare_results(fdtd_results, pstd_results)

    print(f"\n{'=' * 60}")
    if overall_pass:
        print("DIAGNOSTIC PASSED: PSTD behavior within tolerance")
        print("=" * 60)
        return 0
    else:
        print("DIAGNOSTIC FAILED: PSTD shows unexpected behavior")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Check tracing output above for source injection details")
        print("  2. Run with RUST_LOG=kwavers=debug for detailed diagnostics")
        print("  3. Review PSTD_SOURCE_INJECTION_DIAGNOSTIC_2026-02-05.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
