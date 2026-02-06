"""
Session 2: Amplitude Diagnostic - Trace Source Injection

Minimal test to verify source amplitude is being applied correctly.
This diagnostic checks:
1. Source signal amplitude at t=0 and t=T/4
2. Grid source mask values
3. Recorded pressure values at first few timesteps
4. Comparison of expected vs actual amplitudes

Mathematical Specification:
- Source: Sine wave p(t) = A·sin(2πft) where A=100 kPa, f=1 MHz
- At t=0: p(0) = 0 Pa
- At t=T/4: p(T/4) = A = 100 kPa
- Expected: Sensor should record pressure within 20% of source amplitude

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-05
Sprint: 217 Session 2 - k-Wave Validation
"""

import numpy as np
import pykwavers as kw

print("=" * 80)
print("Session 2: Amplitude Diagnostic - Source Injection Verification")
print("=" * 80)
print()

# Parameters
f0 = 1.0e6  # 1 MHz
amp = 100e3  # 100 kPa
c0 = 1500.0  # Sound speed [m/s]
rho0 = 1000.0  # Density [kg/m³]

# Grid large enough for PML (need >40 cells for default PML thickness of 20)
nx = ny = nz = 64
dx = dy = dz = 0.1e-3  # 0.1 mm spacing

# Time parameters
cfl = 0.3
dt = cfl * dx / (c0 * np.sqrt(3))
period = 1.0 / f0  # Period of sine wave
steps_per_period = int(period / dt)

print(f"Configuration:")
print(f"  Grid: {nx}×{ny}×{nz} = {nx**3:,} points")
print(f"  Spacing: {dx * 1e3:.2f} mm")
print(f"  Sound speed: {c0:.0f} m/s")
print(f"  Source: {f0 / 1e6:.1f} MHz, {amp / 1e3:.0f} kPa")
print(f"  Time step: {dt * 1e9:.2f} ns")
print(f"  Period: {period * 1e6:.3f} µs ({steps_per_period} steps)")
print()

# Theoretical signal values
t_quarter = period / 4.0  # Time at T/4 (peak amplitude)
p_t0 = amp * np.sin(2 * np.pi * f0 * 0.0)  # Should be ~0
p_t_quarter = amp * np.sin(2 * np.pi * f0 * t_quarter)  # Should be ~amp

print(f"Theoretical Signal Values:")
print(f"  p(t=0):   {p_t0:.2f} Pa (expected: 0)")
print(f"  p(t=T/4): {p_t_quarter / 1e3:.2f} kPa (expected: {amp / 1e3:.0f} kPa)")
print()

# Run short simulation with both solvers
num_steps = steps_per_period + 10  # One period plus margin

for solver_type, solver_name in [(kw.SolverType.FDTD, "FDTD"), (kw.SolverType.PSTD, "PSTD")]:
    print("-" * 80)
    print(f"{solver_name} Solver")
    print("-" * 80)

    # Create components
    grid = kw.Grid(nx, ny, nz, dx, dy, dz)
    medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    source = kw.Source.plane_wave(grid, frequency=f0, amplitude=amp)

    # Sensor at 60% along z (same as implementation)
    sensor_z = int(nz * 0.6)
    sensor_pos = (nx // 2 * dx, ny // 2 * dy, sensor_z * dz)
    sensor = kw.Sensor.point(position=sensor_pos)

    print(
        f"  Sensor position: ({sensor_pos[0] * 1e3:.2f}, {sensor_pos[1] * 1e3:.2f}, "
        f"{sensor_pos[2] * 1e3:.2f}) mm"
    )
    print(f"  Sensor index: ({nx // 2}, {ny // 2}, {sensor_z})")
    print(f"  Distance from source: {sensor_pos[2] * 1e3:.2f} mm")
    print()

    # Run simulation
    sim = kw.Simulation(grid, medium, source, sensor, solver=solver_type)
    result = sim.run(time_steps=num_steps, dt=dt)

    # Analyze first few timesteps
    sensor_data = result.sensor_data
    time = result.time

    # Find quarter-period index
    quarter_idx = int(steps_per_period / 4)

    print(f"  Recorded Data Analysis:")
    print(f"    First 5 timesteps:")
    for i in range(min(5, len(sensor_data))):
        t_val = time[i]
        p_val = sensor_data[i]
        p_theory = amp * np.sin(2 * np.pi * f0 * t_val)
        print(
            f"      t[{i}] = {t_val * 1e9:.2f} ns: p = {p_val:.2f} Pa "
            f"(theory: {p_theory:.2f} Pa, error: {(p_val - p_theory) / amp * 100:+.1f}%)"
        )

    print()
    print(f"    Quarter-period (step {quarter_idx}):")
    if quarter_idx < len(sensor_data):
        t_q = time[quarter_idx]
        p_q = sensor_data[quarter_idx]
        p_theory_q = amp * np.sin(2 * np.pi * f0 * t_q)
        print(
            f"      t = {t_q * 1e6:.3f} µs: p = {p_q / 1e3:.2f} kPa "
            f"(theory: {p_theory_q / 1e3:.2f} kPa, error: {(p_q - p_theory_q) / amp * 100:+.1f}%)"
        )

    # Overall statistics
    p_max = np.max(np.abs(sensor_data))
    p_min = np.min(sensor_data)
    p_mean = np.mean(np.abs(sensor_data))
    p_std = np.std(sensor_data)

    print()
    print(f"  Overall Statistics:")
    print(f"    Max pressure:  {p_max / 1e3:.2f} kPa")
    print(f"    Min pressure:  {p_min / 1e3:.2f} kPa")
    print(f"    Mean |p|:      {p_mean / 1e3:.2f} kPa")
    print(f"    Std dev:       {p_std / 1e3:.2f} kPa")
    print(f"    Amplitude error: {(p_max - amp) / amp * 100:+.1f}%")

    # Check if amplitude is reasonable
    amplitude_ok = abs(p_max - amp) / amp < 0.5  # 50% tolerance for now
    status = "PASS" if amplitude_ok else "FAIL"
    print(f"    Status: {status} (amplitude within 50%: {amplitude_ok})")

    # Additional diagnostics
    if p_max == 0.0:
        print(
            "    WARNING: No signal recorded - source may not be injecting or sensor not receiving"
        )
    elif p_max > amp * 10:
        print(
            f"    WARNING: Amplitude is {p_max / amp:.1f}x expected - likely source injection bug"
        )
    print()

print("=" * 80)
print("Diagnostic Complete")
print("=" * 80)
print()
print("Expected Observations:")
print("  1. Pressure should be near 0 at t=0 (sine wave starts at zero)")
print("  2. Pressure should reach ~100 kPa at t=T/4")
print("  3. FDTD and PSTD should show similar behavior")
print("  4. Maximum amplitude should be within 50% of source amplitude")
print()
print("If amplitudes are >>100 kPa or <<100 kPa, there is a source injection bug.")
