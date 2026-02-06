"""
Minimal PSTD source injection diagnostic with tracing enabled.

This script creates a simple plane wave source at z=0 and runs PSTD
for a small number of steps with detailed Rust tracing output to
diagnose the source injection amplification bug.

Expected behavior:
- Input signal: 100 kPa sine wave
- FDTD output: ~100 kPa peak
- PSTD output: Currently ~354 kPa (3.54×) - BUG

This script enables RUST_LOG to capture tracing output from the Rust core.
"""

import os
import sys

import numpy as np

# Enable Rust tracing before importing pykwavers
os.environ["RUST_LOG"] = "kwavers=debug"

import pykwavers as kw

# Initialize tracing to enable debug output
kw.init_tracing()


def main():
    """Run minimal PSTD test with plane wave source."""

    # Grid size (must be large enough for PML boundaries)
    nx, ny, nz = 64, 64, 64
    dx = dy = dz = 0.1e-3  # 0.1 mm spacing

    # Temporal parameters (short run)
    c0 = 1500.0  # m/s
    dt = 0.9 * dx / (np.sqrt(3) * c0)  # CFL stability
    nt = 50  # Only 50 steps for quick diagnostics

    print(f"Grid: {nx}×{ny}×{nz}, dt={dt * 1e9:.3f} ns, {nt} steps")
    print(f"CFL number: {c0 * dt / dx:.3f}")
    print("-" * 70)

    # Create grid
    grid = kw.Grid(nx, ny, nz, dx, dy, dz)

    # Homogeneous medium
    medium = kw.Medium.homogeneous(sound_speed=c0, density=1000.0, absorption=0.0)

    # Plane wave source at z=0 boundary
    f0 = 1.0e6  # 1 MHz
    amp = 100e3  # 100 kPa in Pa

    print(f"Source: {f0 / 1e6:.1f} MHz plane wave at z=0")
    print(f"Signal amplitude: {amp / 1e3:.1f} kPa ({amp:.0f} Pa)")

    source = kw.Source.plane_wave(grid, frequency=f0, amplitude=amp)

    # Sensor at center
    sensor_pos = (nx // 2 * dx, ny // 2 * dy, nz // 2 * dz)
    sensor = kw.Sensor.point(position=sensor_pos)

    print(
        f"Sensor position: ({sensor_pos[0] * 1e3:.1f}, {sensor_pos[1] * 1e3:.1f}, {sensor_pos[2] * 1e3:.1f}) mm"
    )
    print("-" * 70)

    # Run with PSTD solver
    print("Running PSTD solver with RUST_LOG=kwavers=debug...")
    print("Watch for tracing output showing source injection details.")
    print("-" * 70)

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    result = sim.run(time_steps=nt, dt=dt)

    # Get sensor data
    pressure = result.sensor_data
    p_max = np.max(np.abs(pressure))
    p_mean = np.mean(np.abs(pressure))

    print("-" * 70)
    print("RESULTS:")
    print(f"Sensor pressure max: {p_max / 1e3:.1f} kPa ({p_max:.0f} Pa)")
    print(f"Sensor pressure mean: {p_mean / 1e3:.3f} kPa")
    print(f"Amplification factor: {p_max / amp:.3f}×")
    print()

    # Expected vs actual
    expected_max = amp  # Should match input amplitude
    error = (p_max - expected_max) / expected_max * 100
    print(f"Expected max: {expected_max / 1e3:.1f} kPa")
    print(f"Error: {error:+.1f}%")
    print()

    if abs(p_max - expected_max) / expected_max > 0.1:
        print("FAILED: Pressure amplitude does not match expected value")
        print("   Check the tracing output above for source injection details:")
        print("   - Is scale=1.0 for boundary plane?")
        print("   - Is is_boundary_plane=true?")
        print("   - What is the contribution value per step?")
        return 1
    else:
        print("PASSED: Pressure amplitude within 10% of expected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
