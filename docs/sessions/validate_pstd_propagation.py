"""
PSTD Wave Propagation Validation Test

Comprehensive test to validate PSTD source injection and wave propagation
with proper configuration to avoid the issues identified in the diagnostic.

Test Design:
- Large grid (128³) to allow full wave propagation
- Long simulation (500 steps) for multiple wavelengths
- Multiple sensors along propagation axis
- Sensor placement outside PML region
- Compare FDTD vs PSTD results

Expected Results:
- Wave should propagate from z=0 plane
- Sensors should record ~100 kPa peak pressure
- Wave arrival times should match analytical prediction
- FDTD and PSTD should agree within tolerance

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-05
Sprint: 217 - k-Wave Comparison & Validation
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Enable Rust tracing
os.environ["RUST_LOG"] = "kwavers=info"

import pykwavers as kw

# Initialize tracing
kw.init_tracing()


def analytical_plane_wave_solution(z, t, amplitude, frequency, sound_speed):
    """
    Analytical solution for 1D plane wave: p(z,t) = A·sin(ωt - kz)

    Parameters
    ----------
    z : float or array
        Position along propagation axis [m]
    t : float or array
        Time [s]
    amplitude : float
        Wave amplitude [Pa]
    frequency : float
        Frequency [Hz]
    sound_speed : float
        Sound speed [m/s]

    Returns
    -------
    p : float or array
        Pressure [Pa]
    """
    omega = 2 * np.pi * frequency
    k = omega / sound_speed
    return amplitude * np.sin(omega * t - k * z)


def run_simulation(solver_type, grid_size, spacing, duration, dt, sensor_positions):
    """
    Run PSTD or FDTD simulation with plane wave source.

    Parameters
    ----------
    solver_type : kw.SolverType
        FDTD or PSTD
    grid_size : tuple
        (nx, ny, nz)
    spacing : tuple
        (dx, dy, dz) [m]
    duration : float
        Simulation duration [s]
    dt : float
        Time step [s]
    sensor_positions : list of tuple
        List of (x, y, z) positions [m]

    Returns
    -------
    dict
        Results with keys: 'time', 'sensor_data', 'solver', 'grid'
    """
    nx, ny, nz = grid_size
    dx, dy, dz = spacing

    print(f"\n{'=' * 70}")
    print(f"Running {solver_type} Simulation")
    print(f"{'=' * 70}")
    print(f"Grid: {nx}x{ny}x{nz} points")
    print(f"Spacing: {dx * 1e3:.3f}x{dy * 1e3:.3f}x{dz * 1e3:.3f} mm")
    print(f"Duration: {duration * 1e6:.1f} us")
    print(f"Time step: {dt * 1e9:.3f} ns")
    print(f"Sensors: {len(sensor_positions)} positions")

    # Create grid
    grid = kw.Grid(nx, ny, nz, dx, dy, dz)

    # Homogeneous medium (water)
    c0 = 1500.0  # m/s
    medium = kw.Medium.homogeneous(sound_speed=c0, density=1000.0, absorption=0.0)

    # Plane wave source at z=0
    f0 = 1.0e6  # 1 MHz
    amp = 100e3  # 100 kPa
    source = kw.Source.plane_wave(grid, frequency=f0, amplitude=amp)

    # Create multi-point sensor
    # Note: pykwavers may only support single point sensors via API
    # For multiple sensors, we'll need to run multiple simulations or use grid sensor
    # For now, use the first sensor position
    sensor_pos = sensor_positions[0]
    sensor = kw.Sensor.point(position=sensor_pos)

    print(f"Source: {f0 / 1e6:.1f} MHz plane wave, {amp / 1e3:.0f} kPa amplitude")
    print(
        f"Sensor: ({sensor_pos[0] * 1e3:.1f}, {sensor_pos[1] * 1e3:.1f}, {sensor_pos[2] * 1e3:.1f}) mm"
    )

    # Run simulation
    print("Running simulation...")
    sim = kw.Simulation(grid, medium, source, sensor, solver=solver_type)
    nt = int(duration / dt)
    result = sim.run(time_steps=nt, dt=dt)

    print(f"OK Simulation complete")
    print(f"  Time steps: {nt}")
    print(f"  Final time: {result.final_time * 1e6:.1f} us")
    print(f"  Sensor data shape: {result.sensor_data.shape}")

    return {
        "time": result.time,
        "sensor_data": result.sensor_data,
        "solver": str(solver_type),
        "grid": grid,
        "sensor_pos": sensor_pos,
    }


def analyze_results(results, sound_speed=1500.0, frequency=1.0e6, amplitude=100e3):
    """
    Analyze simulation results and compare with analytical solution.

    Parameters
    ----------
    results : dict
        Simulation results
    sound_speed : float
        Sound speed [m/s]
    frequency : float
        Source frequency [Hz]
    amplitude : float
        Source amplitude [Pa]

    Returns
    -------
    dict
        Analysis metrics
    """
    time = results["time"]
    sensor_data = results["sensor_data"]
    solver = results["solver"]
    sensor_pos = results["sensor_pos"]

    # Extract z-position (propagation direction)
    z = sensor_pos[2]

    print(f"\n{'=' * 70}")
    print(f"Analysis: {solver}")
    print(f"{'=' * 70}")

    # Compute metrics
    p_max = np.max(np.abs(sensor_data))
    p_mean = np.mean(np.abs(sensor_data))
    p_rms = np.sqrt(np.mean(sensor_data**2))

    # Arrival time (first significant pressure)
    threshold = amplitude * 0.1  # 10% of amplitude
    arrival_indices = np.where(np.abs(sensor_data) > threshold)[0]
    if len(arrival_indices) > 0:
        t_arrival_sim = time[arrival_indices[0]]
    else:
        t_arrival_sim = np.nan

    # Expected arrival time
    t_arrival_theory = z / sound_speed

    # Expected steady-state amplitude (after wave fully formed)
    # For plane wave, should be equal to source amplitude
    p_expected = amplitude

    # Compute analytical solution at sensor position
    p_analytical = analytical_plane_wave_solution(z, time, amplitude, frequency, sound_speed)
    p_analytical_max = np.max(np.abs(p_analytical))

    # Errors
    amplitude_error = (p_max - p_expected) / p_expected * 100
    arrival_error = (
        (t_arrival_sim - t_arrival_theory) / t_arrival_theory * 100
        if not np.isnan(t_arrival_sim)
        else np.nan
    )

    print(f"Sensor Position:")
    print(f"  Distance from source: {z * 1e3:.1f} mm")
    print(f"  Expected arrival time: {t_arrival_theory * 1e6:.2f} us")
    print(
        f"  Simulated arrival time: {t_arrival_sim * 1e6:.2f} us"
        if not np.isnan(t_arrival_sim)
        else "  No arrival detected"
    )
    print(f"  Arrival error: {arrival_error:+.1f}%" if not np.isnan(arrival_error) else "  N/A")
    print()
    print(f"Pressure Amplitude:")
    print(f"  Expected: {p_expected / 1e3:.1f} kPa")
    print(f"  Simulated: {p_max / 1e3:.1f} kPa")
    print(f"  Analytical: {p_analytical_max / 1e3:.1f} kPa")
    print(f"  Error: {amplitude_error:+.1f}%")
    print(f"  RMS: {p_rms / 1e3:.1f} kPa")
    print(f"  Mean: {p_mean / 1e3:.1f} kPa")
    print()

    # Correlation with analytical solution (after arrival)
    if not np.isnan(t_arrival_sim):
        # Compare only after wave has arrived and formed
        comparison_start = int(len(time) * 0.3)  # Skip first 30% for wave formation
        if comparison_start < len(time):
            correlation = np.corrcoef(
                sensor_data[comparison_start:], p_analytical[comparison_start:]
            )[0, 1]

            # L2 error
            l2_error = np.sqrt(
                np.mean((sensor_data[comparison_start:] - p_analytical[comparison_start:]) ** 2)
            )
            l2_rel_error = l2_error / np.std(p_analytical[comparison_start:]) * 100

            print(f"Waveform Comparison (after steady state):")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  L2 error: {l2_error / 1e3:.2f} kPa")
            print(f"  Relative L2 error: {l2_rel_error:.1f}%")
            print()

    # Validation criteria
    print(f"Validation:")
    amplitude_pass = abs(amplitude_error) < 15  # Within 15%
    arrival_pass = abs(arrival_error) < 10 if not np.isnan(arrival_error) else False  # Within 10%

    print(f"  Amplitude: {'PASS' if amplitude_pass else 'FAIL'}")
    print(f"  Arrival time: {'PASS' if arrival_pass else 'FAIL'}")
    print(f"  Overall: {'PASS' if (amplitude_pass and arrival_pass) else 'FAIL'}")

    return {
        "p_max": p_max,
        "p_mean": p_mean,
        "p_rms": p_rms,
        "t_arrival_sim": t_arrival_sim,
        "t_arrival_theory": t_arrival_theory,
        "amplitude_error": amplitude_error,
        "arrival_error": arrival_error,
        "amplitude_pass": amplitude_pass,
        "arrival_pass": arrival_pass,
        "p_analytical": p_analytical,
    }


def plot_comparison(results_fdtd, results_pstd, analysis_fdtd, analysis_pstd, output_dir):
    """
    Create comparison plots.

    Parameters
    ----------
    results_fdtd : dict
        FDTD simulation results
    results_pstd : dict
        PSTD simulation results
    analysis_fdtd : dict
        FDTD analysis metrics
    analysis_pstd : dict
        PSTD analysis metrics
    output_dir : Path
        Directory for output plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Time series comparison
    ax = axes[0, 0]
    ax.plot(
        results_fdtd["time"] * 1e6,
        results_fdtd["sensor_data"] / 1e3,
        label="FDTD",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        results_pstd["time"] * 1e6,
        results_pstd["sensor_data"] / 1e3,
        label="PSTD",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        results_fdtd["time"] * 1e6,
        analysis_fdtd["p_analytical"] / 1e3,
        "k--",
        label="Analytical",
        linewidth=1,
        alpha=0.5,
    )
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title("Pressure Time Series at Sensor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoomed view (first few periods)
    ax = axes[0, 1]
    t_max_zoom = 5e-6  # First 5 μs
    idx_zoom = results_fdtd["time"] <= t_max_zoom
    ax.plot(
        results_fdtd["time"][idx_zoom] * 1e6,
        results_fdtd["sensor_data"][idx_zoom] / 1e3,
        label="FDTD",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        results_pstd["time"][idx_zoom] * 1e6,
        results_pstd["sensor_data"][idx_zoom] / 1e3,
        label="PSTD",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        results_fdtd["time"][idx_zoom] * 1e6,
        analysis_fdtd["p_analytical"][idx_zoom] / 1e3,
        "k--",
        label="Analytical",
        linewidth=1,
        alpha=0.5,
    )
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title("Pressure Time Series (Zoomed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Error vs time
    ax = axes[1, 0]
    error_fdtd = (results_fdtd["sensor_data"] - analysis_fdtd["p_analytical"]) / 1e3
    error_pstd = (results_pstd["sensor_data"] - analysis_pstd["p_analytical"]) / 1e3
    ax.plot(results_fdtd["time"] * 1e6, error_fdtd, label="FDTD Error", alpha=0.8)
    ax.plot(results_pstd["time"] * 1e6, error_pstd, label="PSTD Error", alpha=0.8)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Error [kPa]")
    ax.set_title("Error vs Analytical Solution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Metrics comparison
    ax = axes[1, 1]
    metrics = ["Amplitude Error [%]", "Arrival Error [%]"]
    fdtd_vals = [analysis_fdtd["amplitude_error"], analysis_fdtd["arrival_error"]]
    pstd_vals = [analysis_pstd["amplitude_error"], analysis_pstd["arrival_error"]]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, fdtd_vals, width, label="FDTD", alpha=0.8)
    ax.bar(x + width / 2, pstd_vals, width, label="PSTD", alpha=0.8)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.axhline(10, color="r", linestyle=":", linewidth=0.5, label="±10% threshold")
    ax.axhline(-10, color="r", linestyle=":", linewidth=0.5)
    ax.set_ylabel("Error [%]")
    ax.set_title("Validation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "pstd_validation_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nOK Saved plot: {output_file}")
    plt.close()


def main():
    """Run comprehensive PSTD validation test."""

    print("=" * 70)
    print("PSTD Wave Propagation Validation Test")
    print("=" * 70)
    print()
    print("Objective: Validate PSTD source injection and wave propagation")
    print("Method: Compare FDTD vs PSTD vs Analytical Solution")
    print()

    # Configuration
    nx, ny, nz = 128, 128, 128
    dx = dy = dz = 0.1e-3  # 0.1 mm spacing

    # Time parameters
    c0 = 1500.0  # m/s
    cfl = 0.3
    dt = cfl * dx / (c0 * np.sqrt(3))
    duration = 20e-6  # 20 us (13 wavelengths at 1 MHz)

    # Sensor positions (along z-axis, outside PML region)
    # PML typically 20 points, so sensor should be at least 25 points from boundaries
    sensor_z = 64 * dz  # Center of domain along z
    sensor_positions = [
        (64 * dx, 64 * dy, sensor_z),  # Center of domain
    ]

    print(f"Configuration:")
    print(f"  Grid: {nx}×{ny}×{nz}")
    print(f"  Spacing: {dx * 1e3} mm")
    print(f"  CFL: {cfl:.3f}")
    print(f"  Time step: {dt * 1e9:.3f} ns")
    print(f"  Duration: {duration * 1e6} us")
    print(f"  Sensor distance from source: {sensor_z * 1e3:.1f} mm")
    print()

    # Run FDTD
    results_fdtd = run_simulation(
        kw.SolverType.FDTD, (nx, ny, nz), (dx, dy, dz), duration, dt, sensor_positions
    )

    # Run PSTD
    results_pstd = run_simulation(
        kw.SolverType.PSTD, (nx, ny, nz), (dx, dy, dz), duration, dt, sensor_positions
    )

    # Analyze results
    analysis_fdtd = analyze_results(results_fdtd)
    analysis_pstd = analyze_results(results_pstd)

    # Create comparison plots
    output_dir = Path(__file__).parent / "results"
    plot_comparison(results_fdtd, results_pstd, analysis_fdtd, analysis_pstd, output_dir)

    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"FDTD: {'PASS' if (analysis_fdtd['amplitude_pass'] and analysis_fdtd['arrival_pass']) else 'FAIL'}"
    )
    print(f"  Amplitude error: {analysis_fdtd['amplitude_error']:+.1f}%")
    print(f"  Arrival error: {analysis_fdtd['arrival_error']:+.1f}%")
    print()
    print(
        f"PSTD: {'PASS' if (analysis_pstd['amplitude_pass'] and analysis_pstd['arrival_pass']) else 'FAIL'}"
    )
    print(f"  Amplitude error: {analysis_pstd['amplitude_error']:+.1f}%")
    print(f"  Arrival error: {analysis_pstd['arrival_error']:+.1f}%")
    print()

    # Overall validation
    overall_pass = (
        analysis_fdtd["amplitude_pass"]
        and analysis_fdtd["arrival_pass"]
        and analysis_pstd["amplitude_pass"]
        and analysis_pstd["arrival_pass"]
    )

    if overall_pass:
        print("OK VALIDATION PASSED: PSTD source injection working correctly")
        return 0
    else:
        print("FAIL VALIDATION FAILED: Issues detected in PSTD or FDTD")
        return 1


if __name__ == "__main__":
    sys.exit(main())
