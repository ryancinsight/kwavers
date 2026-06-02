#!/usr/bin/env python3
"""
Plane Wave Comparison: k-Wave vs pykwavers

This example demonstrates identical plane wave simulations in both k-Wave
(MATLAB/Python) and pykwavers to enable direct validation and performance comparison.

Mathematical Specification:
- Problem: Plane wave propagation in homogeneous medium
- Domain: 64×64×64 grid, 0.1 mm spacing (6.4×6.4×6.4 mm)
- Medium: Water (c=1500 m/s, ρ=1000 kg/m³)
- Source: 1 MHz plane wave, 100 kPa amplitude, +z propagation
- Duration: 10 μs (15 wavelengths @ 1500 m/s)
- Sensor: Central point (32, 32, 32)

Expected Results:
- Wavelength: λ = c/f = 1500/1e6 = 1.5 mm = 15 grid points
- Wave arrival: t = z/c = (32*0.1e-3)/1500 ≈ 2.13 μs
- Pressure amplitude: ~100 kPa (with numerical dispersion)

References:
1. Treeby & Cox (2010). k-Wave MATLAB toolbox. J. Biomed. Opt.
2. kwavers ARCHITECTURE.md
3. Sprint 217 Session 9 - Python Integration Specification

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-04
Sprint: 217 Session 9 - k-Wave Comparison via PyO3
"""

import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# pykwavers (Rust-backed)
import pykwavers as kw

# k-Wave bridge (MATLAB Engine - optional)
try:
    from pykwavers.kwave_bridge import (
        MATLAB_AVAILABLE,
        GridConfig,
        KWaveBridge,
        MediumConfig,
        SensorConfig,
        SourceConfig,
    )

    KWAVE_AVAILABLE = MATLAB_AVAILABLE
except ImportError:
    KWAVE_AVAILABLE = False
    warnings.warn("k-Wave bridge not available. Comparison will use pykwavers only.", UserWarning)


# ============================================================================
# Simulation Parameters (Identical for Both)
# ============================================================================

# Grid configuration
NX, NY, NZ = 64, 64, 64
DX, DY, DZ = 0.1e-3, 0.1e-3, 0.1e-3  # 0.1 mm spacing

# Medium properties (water at 20°C)
SOUND_SPEED = 1500.0  # m/s
DENSITY = 1000.0  # kg/m³
ABSORPTION = 0.0  # dB/(MHz·cm) - no absorption for analytical comparison

# Source parameters
FREQUENCY = 1e6  # 1 MHz
AMPLITUDE = 1e5  # 100 kPa
WAVELENGTH = SOUND_SPEED / FREQUENCY  # 1.5 mm = 15 grid points

# Time parameters
DURATION = 10e-6  # 10 μs
CFL = 0.3  # Conservative CFL number
DT = CFL * DX / SOUND_SPEED  # Time step from CFL condition
TIME_STEPS = int(DURATION / DT)

# Sensor location (center of domain)
SENSOR_POS = (NX // 2 * DX, NY // 2 * DY, NZ // 2 * DZ)

print("=" * 80)
print("Plane Wave Comparison: k-Wave vs pykwavers")
print("=" * 80)
print(f"\nGrid: {NX}×{NY}×{NZ} points, {DX * 1e3:.2f}×{DY * 1e3:.2f}×{DZ * 1e3:.2f} mm spacing")
print(f"Domain: {NX * DX * 1e3:.2f}×{NY * DY * 1e3:.2f}×{NZ * DZ * 1e3:.2f} mm")
print(f"Medium: Water (c={SOUND_SPEED} m/s, ρ={DENSITY} kg/m³)")
print(f"Source: {FREQUENCY * 1e-6:.1f} MHz plane wave, {AMPLITUDE * 1e-3:.0f} kPa amplitude")
print(f"Wavelength: λ = {WAVELENGTH * 1e3:.2f} mm = {WAVELENGTH / DX:.1f} grid points")
print(f"Duration: {DURATION * 1e6:.1f} μs ({TIME_STEPS} steps, dt={DT * 1e9:.2f} ns)")
print(
    f"Sensor: Position = ({SENSOR_POS[0] * 1e3:.2f}, {SENSOR_POS[1] * 1e3:.2f}, {SENSOR_POS[2] * 1e3:.2f}) mm"
)
print()


# ============================================================================
# pykwavers Simulation (Rust-backed)
# ============================================================================


def run_pykwavers() -> Dict[str, np.ndarray]:
    """
    Run plane wave simulation using pykwavers (Rust backend).

    Returns
    -------
    dict
        Results dictionary with keys:
        - 'pressure': Pressure time series at sensor [Pa]
        - 'time': Time vector [s]
        - 'runtime': Execution time [s]
    """
    print("Running pykwavers simulation...")
    start_time = time.perf_counter()

    # Create grid
    grid = kw.Grid(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ)

    # Create medium
    medium = kw.Medium.homogeneous(sound_speed=SOUND_SPEED, density=DENSITY, absorption=ABSORPTION)

    # Create source
    source = kw.Source.plane_wave(
        grid=grid,
        frequency=FREQUENCY,
        amplitude=AMPLITUDE,
        direction=(0, 0, 1),  # +z propagation
    )

    # Create sensor
    sensor = kw.Sensor.point(position=SENSOR_POS)

    # Run simulation
    sim = kw.Simulation(grid, medium, source, sensor)
    result = sim.run(time_steps=TIME_STEPS, dt=DT)

    runtime = time.perf_counter() - start_time

    # Extract results (placeholder - real implementation would return time series)
    time_vec = np.arange(TIME_STEPS) * DT
    pressure = np.zeros(TIME_STEPS)  # Placeholder: will be populated by real simulation

    print(f"  ✓ Completed in {runtime:.3f} seconds")
    print(f"  ✓ Grid size: {grid.total_points():,} points")
    print(f"  ✓ Time steps: {TIME_STEPS:,}")
    print(f"  ✓ Final time: {result.final_time * 1e6:.2f} μs")
    print()

    return {
        "pressure": pressure,
        "time": time_vec,
        "runtime": runtime,
        "method": "pykwavers (Rust FDTD)",
    }


# ============================================================================
# k-Wave Simulation (MATLAB Engine)
# ============================================================================


def run_kwave() -> Optional[Dict[str, np.ndarray]]:
    """
    Run plane wave simulation using k-Wave (MATLAB Engine).

    Returns
    -------
    dict or None
        Results dictionary (same structure as pykwavers) or None if unavailable
    """
    if not KWAVE_AVAILABLE:
        print("k-Wave unavailable (MATLAB Engine not found)")
        print()
        return None

    print("Running k-Wave simulation...")
    start_time = time.perf_counter()

    try:
        with KWaveBridge() as bridge:
            # Grid configuration
            grid_config = GridConfig(Nx=NX, Ny=NY, Nz=NZ, dx=DX, dy=DY, dz=DZ, pml_size=10)

            # Medium configuration
            medium_config = MediumConfig(
                sound_speed=SOUND_SPEED, density=DENSITY, alpha_coeff=ABSORPTION, alpha_power=1.5
            )

            # Source configuration (plane wave at z=0)
            source_mask = np.zeros((NX, NY, NZ), dtype=bool)
            source_mask[:, :, 0] = True
            time_vec = np.arange(TIME_STEPS) * DT
            source_signal = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * time_vec)

            source_config = SourceConfig(
                p_mask=source_mask, p=source_signal, source_type="pressure"
            )

            # Sensor configuration (point at center)
            sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
            sensor_mask[NX // 2, NY // 2, NZ // 2] = True

            sensor_config = SensorConfig(mask=sensor_mask, record=["p"])

            # Run simulation
            result = bridge.run_simulation(
                grid_config,
                medium_config,
                source_config,
                sensor_config,
                time_steps=TIME_STEPS,
                dt=DT,
            )

            runtime = time.perf_counter() - start_time

            print(f"  ✓ Completed in {runtime:.3f} seconds")
            print(f"  ✓ Sensor data shape: {result.sensor_data.shape}")
            print()

            return {
                "pressure": result.sensor_data.flatten()
                if result.sensor_data.ndim > 1
                else result.sensor_data,
                "time": time_vec,
                "runtime": runtime,
                "method": "k-Wave (MATLAB PSTD)",
            }

    except Exception as e:
        print(f"  ✗ k-Wave simulation failed: {e}")
        print()
        return None


# ============================================================================
# Comparison and Visualization
# ============================================================================


def compare_results(pykwavers_results: Dict, kwave_results: Optional[Dict]) -> None:
    """
    Compare and visualize results from both simulators.

    Parameters
    ----------
    pykwavers_results : dict
        Results from pykwavers
    kwave_results : dict or None
        Results from k-Wave (None if unavailable)
    """
    print("=" * 80)
    print("Comparison Results")
    print("=" * 80)
    print()

    # Performance comparison
    print(f"pykwavers runtime: {pykwavers_results['runtime']:.3f} seconds")
    if kwave_results:
        print(f"k-Wave runtime:    {kwave_results['runtime']:.3f} seconds")
        speedup = kwave_results["runtime"] / pykwavers_results["runtime"]
        print(
            f"Speedup:           {speedup:.2f}x {'(pykwavers faster)' if speedup > 1 else '(k-Wave faster)'}"
        )
    print()

    # Error metrics (if both available)
    if kwave_results:
        p_kwavers = pykwavers_results["pressure"]
        p_kwave = kwave_results["pressure"]

        # Ensure same length
        min_len = min(len(p_kwavers), len(p_kwave))
        p_kwavers = p_kwavers[:min_len]
        p_kwave = p_kwave[:min_len]

        # Compute error metrics
        l2_error = np.linalg.norm(p_kwavers - p_kwave) / np.linalg.norm(p_kwave)
        linf_error = np.max(np.abs(p_kwavers - p_kwave)) / np.max(np.abs(p_kwave))
        rmse = np.sqrt(np.mean((p_kwavers - p_kwave) ** 2))

        print(f"Relative L2 error:   {l2_error:.2e}")
        print(f"Relative L∞ error:   {linf_error:.2e}")
        print(f"RMSE:                {rmse:.2e} Pa")
        print()

        # Validation against acceptance criteria (from Sprint 217 specs)
        print("Validation Status:")
        print(f"  L2 < 0.01:  {'✓ PASS' if l2_error < 0.01 else '✗ FAIL'}")
        print(f"  L∞ < 0.05:  {'✓ PASS' if linf_error < 0.05 else '✗ FAIL'}")
        print()

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Pressure time series
    ax1 = axes[0]
    time_us = pykwavers_results["time"] * 1e6
    ax1.plot(time_us, pykwavers_results["pressure"] / 1e3, "b-", label="pykwavers", linewidth=2)
    if kwave_results:
        ax1.plot(
            time_us[: len(kwave_results["pressure"])],
            kwave_results["pressure"] / 1e3,
            "r--",
            label="k-Wave",
            linewidth=1.5,
            alpha=0.7,
        )
    ax1.set_xlabel("Time [μs]")
    ax1.set_ylabel("Pressure [kPa]")
    ax1.set_title("Plane Wave Pressure at Sensor")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error (if k-Wave available)
    ax2 = axes[1]
    if kwave_results:
        error = (pykwavers_results["pressure"][:min_len] - kwave_results["pressure"]) / 1e3
        ax2.plot(time_us[:min_len], error, "k-", linewidth=1.5)
        ax2.set_xlabel("Time [μs]")
        ax2.set_ylabel("Pressure Error [kPa]")
        ax2.set_title("pykwavers - k-Wave Difference")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="r", linestyle="--", linewidth=1, alpha=0.5)
    else:
        ax2.text(
            0.5,
            0.5,
            "k-Wave comparison unavailable\n(MATLAB Engine not installed)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=14,
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "plane_wave_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")

    plt.show()


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main comparison workflow."""
    # Run pykwavers simulation
    pykwavers_results = run_pykwavers()

    # Run k-Wave simulation (if available)
    kwave_results = run_kwave()

    # Compare and visualize
    compare_results(pykwavers_results, kwave_results)

    print("=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
