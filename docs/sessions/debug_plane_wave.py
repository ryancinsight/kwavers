#!/usr/bin/env python3
"""
Plane Wave Source Debug Diagnostics

This script investigates the timing error by:
1. Verifying the source mask is boundary-only
2. Visualizing early-time pressure field evolution
3. Analyzing wavefront propagation characteristics

Author: Ryan Clanton (@ryancinsight)
Date: 2024-02-04
Sprint: 217 Session 9 - Phase 4 Development
"""

import matplotlib.pyplot as plt
import numpy as np
import pykwavers as kw


def analyze_mask_application():
    """
    Test to verify that boundary-only injection is working.

    This indirectly tests by running a very short simulation and checking
    if pressure appears throughout the grid (FullGrid mode) or only near
    the boundary (BoundaryOnly mode).
    """
    print("=" * 70)
    print("Mask Application Analysis")
    print("=" * 70)

    # Small grid for analysis
    grid = kw.Grid(nx=16, ny=16, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0, absorption=0.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5, direction=(0.0, 0.0, 1.0))

    # Sensor at boundary (z=0)
    sensor_boundary = kw.Sensor.point(position=(0.8e-3, 0.8e-3, 0.0))

    # Run for just 5 time steps
    sim = kw.Simulation(grid, medium, source, sensor_boundary)
    dt = 0.3 * 0.1e-3 / 1500.0
    result = sim.run(time_steps=5, dt=dt)

    print(f"\nBoundary sensor (z=0) after 5 steps:")
    print(f"  Max pressure: {np.max(np.abs(result.sensor_data)):.2e} Pa")
    print(
        f"  First non-zero at step: {np.where(np.abs(result.sensor_data) > 1e-10)[0][0] if np.any(np.abs(result.sensor_data) > 1e-10) else 'none'}"
    )

    # Sensor deep in domain
    sensor_deep = kw.Sensor.point(position=(0.8e-3, 0.8e-3, 1.6e-3))
    sim_deep = kw.Simulation(grid, medium, source, sensor_deep)
    result_deep = sim_deep.run(time_steps=5, dt=dt)

    print(f"\nDeep sensor (z=1.6mm) after 5 steps:")
    print(f"  Max pressure: {np.max(np.abs(result_deep.sensor_data)):.2e} Pa")
    print(f"  Expected arrival: {1.6e-3 / 1500.0 * 1e6:.2f} us")
    print(f"  Actual time elapsed: {5 * dt * 1e6:.2f} us")

    if np.max(np.abs(result_deep.sensor_data)) > 1e-3:
        print("  WARNING: Wave reached deep sensor too quickly!")
        print("     This suggests FullGrid mode (spatial pre-population)")
    else:
        print("  OK: No significant pressure at deep sensor")
        print("     This suggests BoundaryOnly mode is active")


def analyze_wavefront_propagation():
    """
    Analyze wavefront position vs time to measure effective wave speed.
    """
    print("\n" + "=" * 70)
    print("Wavefront Propagation Analysis")
    print("=" * 70)

    grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0, absorption=0.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

    # Multiple sensors at different depths
    depths_mm = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"\n{'Depth (mm)':>12} {'Expected (us)':>15} {'Measured (us)':>15} {'Error (%)':>12}")
    print("-" * 70)

    arrival_times = []
    expected_times = []

    for depth_mm in depths_mm:
        depth_m = depth_mm * 1e-3
        z_idx = int(depth_m / 0.1e-3)

        sensor_pos = (3.2e-3, 3.2e-3, depth_m)
        sensor = kw.Sensor.point(position=sensor_pos)

        expected_time = depth_m / 1500.0
        expected_times.append(expected_time)

        sim = kw.Simulation(grid, medium, source, sensor)

        duration = 3.0 * expected_time
        dt = 0.3 * 0.1e-3 / 1500.0
        time_steps = int(duration / dt)

        result = sim.run(time_steps=time_steps, dt=dt)

        # Find arrival with 10% threshold
        p_max = np.max(np.abs(result.sensor_data))
        threshold = 0.1 * p_max
        crossings = np.where(np.abs(result.sensor_data) > threshold)[0]

        if len(crossings) > 0:
            measured_time = result.time[crossings[0]]
            arrival_times.append(measured_time)
            error = (measured_time - expected_time) / expected_time * 100

            print(
                f"{depth_mm:>12.1f} {expected_time * 1e6:>15.3f} {measured_time * 1e6:>15.3f} {error:>12.1f}"
            )
        else:
            arrival_times.append(np.nan)
            print(f"{depth_mm:>12.1f} {expected_time * 1e6:>15.3f} {'NO ARRIVAL':>15} {'N/A':>12}")

    # Fit linear relationship to extract effective speed
    valid_indices = ~np.isnan(arrival_times)
    if np.sum(valid_indices) >= 2:
        depths_m = np.array([d * 1e-3 for d in depths_mm])
        fit = np.polyfit(depths_m[valid_indices], np.array(arrival_times)[valid_indices], 1)
        effective_speed = 1.0 / fit[0]  # Speed from slope
        time_offset = fit[1]  # y-intercept (initialization delay)

        print(f"\nLinear Fit Results:")
        print(f"  Effective speed: {effective_speed:.1f} m/s (expected: 1500 m/s)")
        print(f"  Time offset: {time_offset * 1e6:.3f} us (initialization delay)")
        print(f"  Speed error: {(effective_speed - 1500) / 1500 * 100:.1f}%")


def analyze_source_buildup():
    """
    Examine pressure evolution at the source boundary to understand wave initialization.
    """
    print("\n" + "=" * 70)
    print("Source Boundary Buildup Analysis")
    print("=" * 70)

    grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0, absorption=0.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

    # Sensor right at the boundary
    sensor = kw.Sensor.point(position=(1.6e-3, 1.6e-3, 0.05e-3))  # One cell from boundary

    sim = kw.Simulation(grid, medium, source, sensor)

    # Run for 1 period
    period = 1.0 / 1e6
    dt = 0.3 * 0.1e-3 / 1500.0
    time_steps = int(2 * period / dt)

    result = sim.run(time_steps=time_steps, dt=dt)

    # Analyze first few cycles
    print(f"\nFirst {time_steps} time steps ({time_steps * dt * 1e6:.2f} us):")
    print(f"  Time step: {dt * 1e9:.3f} ns")
    print(f"  Expected period: {period * 1e6:.3f} us")
    print(f"  Expected amplitude: {1e5:.2e} Pa")
    print(f"  Measured max amplitude: {np.max(np.abs(result.sensor_data)):.2e} Pa")

    # Find when amplitude reaches 50% of expected
    half_amplitude = 0.5 * 1e5
    half_crossing = np.where(np.abs(result.sensor_data) > half_amplitude)[0]
    if len(half_crossing) > 0:
        buildup_time = result.time[half_crossing[0]]
        print(
            f"  Time to 50% amplitude: {buildup_time * 1e6:.3f} us ({buildup_time / dt:.0f} steps)"
        )


def visualize_early_time_evolution():
    """
    Create visualization of pressure field at early times.
    """
    print("\n" + "=" * 70)
    print("Generating Early-Time Pressure Evolution Visualization")
    print("=" * 70)

    # Run simulation with sensors at multiple z positions
    grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0, absorption=0.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

    # Sample at different z positions
    z_positions = np.linspace(0, 3.1e-3, 16)

    # Run simulation for each position
    dt = 0.3 * 0.1e-3 / 1500.0
    duration = 4e-6  # 4 microseconds
    time_steps = int(duration / dt)

    pressure_vs_z_t = []

    print(f"\nRunning {len(z_positions)} simulations to map pressure field...")

    for z_pos in z_positions:
        sensor = kw.Sensor.point(position=(1.6e-3, 1.6e-3, z_pos))
        sim = kw.Simulation(grid, medium, source, sensor)
        result = sim.run(time_steps=time_steps, dt=dt)
        pressure_vs_z_t.append(result.sensor_data)

    pressure_field = np.array(pressure_vs_z_t)  # Shape: (nz, nt)
    time_vec = np.arange(time_steps) * dt

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Space-time diagram
    ax = axes[0, 0]
    extent = [time_vec[0] * 1e6, time_vec[-1] * 1e6, z_positions[0] * 1e3, z_positions[-1] * 1e3]
    im = ax.imshow(
        pressure_field / 1e5,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    # Overlay theoretical wavefront
    c = 1500.0
    z_theory = c * time_vec
    ax.plot(time_vec * 1e6, z_theory * 1e3, "k--", linewidth=2, label="Theory: z = ct")

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Pressure Field Evolution\n(Space-Time Diagram)")
    ax.legend()
    plt.colorbar(im, ax=ax, label="Pressure (100 kPa)")

    # Plot 2: Snapshots at different times
    ax = axes[0, 1]
    snapshot_times_us = [0.5, 1.0, 1.5, 2.0]
    for t_us in snapshot_times_us:
        t_idx = int(t_us * 1e-6 / dt)
        if t_idx < time_steps:
            ax.plot(z_positions * 1e3, pressure_field[:, t_idx] / 1e5, label=f"t = {t_us:.1f} us")
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Pressure (100 kPa)")
    ax.set_title("Pressure Profiles at Different Times")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)

    # Plot 3: Wavefront position vs time
    ax = axes[1, 0]
    # Extract wavefront position (50% amplitude threshold)
    threshold = 0.5 * np.max(np.abs(pressure_field))
    wavefront_positions = []
    wavefront_times = []

    for t_idx in range(time_steps):
        p_slice = np.abs(pressure_field[:, t_idx])
        crossings = np.where(p_slice > threshold)[0]
        if len(crossings) > 0:
            # Find furthest crossing (wavefront)
            z_idx = crossings[-1]
            wavefront_positions.append(z_positions[z_idx])
            wavefront_times.append(time_vec[t_idx])

    if len(wavefront_positions) > 0:
        ax.plot(
            np.array(wavefront_times) * 1e6,
            np.array(wavefront_positions) * 1e3,
            "b.",
            markersize=3,
            label="Measured wavefront",
        )

        # Theoretical wavefront
        ax.plot(time_vec * 1e6, c * time_vec * 1e3, "r--", linewidth=2, label="Theory: z = ct")

        # Fit measured wavefront
        if len(wavefront_positions) >= 2:
            fit = np.polyfit(wavefront_times, wavefront_positions, 1)
            fitted_speed = fit[0]
            offset = fit[1]
            ax.plot(
                time_vec * 1e6,
                (fit[0] * time_vec + fit[1]) * 1e3,
                "g-",
                linewidth=1.5,
                label=f"Fit: c={fitted_speed:.0f} m/s, offset={offset * 1e3:.2f} mm",
            )

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Wavefront Position (mm)")
    ax.set_title("Wavefront Propagation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Time series at specific positions
    ax = axes[1, 1]
    positions_mm = [0.5, 1.0, 1.5, 2.0]
    for pos_mm in positions_mm:
        z_idx = int(pos_mm * 1e-3 / (z_positions[1] - z_positions[0]))
        if z_idx < len(z_positions):
            expected_arrival = pos_mm * 1e-3 / c
            ax.plot(time_vec * 1e6, pressure_field[z_idx, :] / 1e5, label=f"z = {pos_mm:.1f} mm")
            ax.axvline(
                expected_arrival * 1e6, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
            )

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Pressure (100 kPa)")
    ax.set_title("Pressure Time Series at Different Depths")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("plane_wave_debug_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization: plane_wave_debug_analysis.png")

    plt.show()


def main():
    """Run all diagnostic analyses."""
    print("\n" + "=" * 70)
    print("PLANE WAVE SOURCE DEBUG DIAGNOSTICS")
    print("=" * 70)
    print("\nThis script investigates the ~24% timing error by:")
    print("1. Verifying boundary-only source injection")
    print("2. Measuring effective wave speed")
    print("3. Analyzing source initialization buildup")
    print("4. Visualizing wavefront propagation")
    print()

    try:
        analyze_mask_application()
        analyze_wavefront_propagation()
        analyze_source_buildup()
        visualize_early_time_evolution()

        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print("\nCheck the generated visualization 'plane_wave_debug_analysis.png'")
        print("for detailed space-time evolution of the pressure field.")
        print("\nLook for:")
        print("  - Wavefront slope (should match c = 1500 m/s)")
        print("  - Time offset at z=0 (initialization delay)")
        print("  - Any precursor waves or numerical artifacts")
        print()

    except Exception as e:
        print(f"\nError during diagnostics: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
