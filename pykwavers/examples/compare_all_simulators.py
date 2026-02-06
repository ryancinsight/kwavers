#!/usr/bin/env python3
"""
Complete Three-Way Comparison: pykwavers, k-wave-python, k-Wave MATLAB

This example demonstrates comprehensive comparison across all available
simulators with automated validation and visualization.

Mathematical Specification:
- Problem: Plane wave propagation in homogeneous medium
- Domain: 64×64×64 grid, 0.1 mm spacing (6.4×6.4×6.4 mm)
- Medium: Water (c=1500 m/s, rho=1000 kg/m^3)
- Source: 1 MHz plane wave, 100 kPa amplitude, +z propagation
- Duration: 10 us (15 wavelengths @ 1500 m/s)
- Sensor: Central point (32, 32, 32)

Validation Criteria:
- L2 error < 0.01 (1% relative error)
- L∞ error < 0.05 (5% relative error)
- Correlation > 0.99

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-04
Sprint: 217 Session 10 - k-wave-python Integration
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from pykwavers.comparison import (
        ComparisonResult,
        SimulationConfig,
        SimulatorType,
        plot_comparison,
        run_comparison,
    )

    COMPARISON_AVAILABLE = True
except ImportError as e:
    print(f"[X] Comparison framework not available: {e}")
    COMPARISON_AVAILABLE = False

try:
    import pykwavers as kw

    PYKWAVERS_AVAILABLE = True
except ImportError:
    print("[X] pykwavers not available. Build with: maturin develop --release")
    PYKWAVERS_AVAILABLE = False

try:
    from pykwavers.kwave_python_bridge import KWAVE_PYTHON_AVAILABLE
except ImportError:
    KWAVE_PYTHON_AVAILABLE = False

try:
    from pykwavers.kwave_bridge import MATLAB_AVAILABLE
except ImportError:
    MATLAB_AVAILABLE = False


def print_header():
    """Print comparison header."""
    print("=" * 80)
    print("Complete Three-Way Comparison")
    print("pykwavers (Rust) vs k-wave-python (C++) vs k-Wave (MATLAB)")
    print("=" * 80)
    print()


def check_availability():
    """Check and report available simulators."""
    print("Checking available simulators...")
    print()
    print(
        f"  pykwavers (Rust):         {'[OK] Available' if PYKWAVERS_AVAILABLE else '[X] Not found'}"
    )
    print(
        f"  k-wave-python (C++):      {'[OK] Available' if KWAVE_PYTHON_AVAILABLE else '[X] Not found'}"
    )
    print(
        f"  k-Wave MATLAB:            {'[OK] Available' if MATLAB_AVAILABLE else '[X] Not found'}"
    )
    print(
        f"  Comparison framework:     {'[OK] Available' if COMPARISON_AVAILABLE else '[X] Not found'}"
    )
    print()

    if not PYKWAVERS_AVAILABLE:
        print("ERROR: pykwavers is required. Install with:")
        print("  cd kwavers/pykwavers")
        print("  maturin develop --release")
        return False

    if not COMPARISON_AVAILABLE:
        print("ERROR: Comparison framework not available.")
        return False

    if not KWAVE_PYTHON_AVAILABLE and not MATLAB_AVAILABLE:
        print("WARNING: No k-Wave implementation available for comparison.")
        print("Install k-wave-python with: pip install k-wave-python")
        print("Or install MATLAB Engine API for k-Wave MATLAB bridge.")
        print()
        print("Continuing with pykwavers-only simulation...")

    return True


def create_config() -> SimulationConfig:
    """
    Create standard test configuration.

    Returns:
        SimulationConfig for plane wave in water
    """
    print("Configuration:")
    print("-" * 80)

    config = SimulationConfig(
        # Grid (64³, 0.1 mm spacing)
        grid_shape=(64, 64, 64),
        grid_spacing=(0.1e-3, 0.1e-3, 0.1e-3),  # [m]
        # Medium (water at 20°C)
        sound_speed=1500.0,  # [m/s]
        density=1000.0,  # [kg/m³]
        # Source (1 MHz plane wave, 100 kPa)
        source_frequency=1e6,  # [Hz]
        source_amplitude=1e5,  # [Pa]
        source_position=None,  # None = plane wave
        # Sensor (center point)
        sensor_position=(3.2e-3, 3.2e-3, 3.2e-3),  # [m]
        # Time
        duration=10e-6,  # [s]
        # Boundary
        pml_size=10,
        cfl=0.3,
    )

    print(f"  Grid:       {config.grid_shape}")
    print(f"  Spacing:    {tuple(s * 1e3 for s in config.grid_spacing)} mm")
    print(
        f"  Domain:     {tuple(s * c * 1e3 for s, c in zip(config.grid_spacing, config.grid_shape))} mm"
    )
    print(f"  Medium:     c={config.sound_speed} m/s, rho={config.density} kg/m^3")
    print(
        f"  Source:     {config.source_frequency * 1e-6:.1f} MHz, {config.source_amplitude * 1e-3:.0f} kPa"
    )
    print(
        f"  Wavelength: {config.wavelength * 1e3:.2f} mm ({config.points_per_wavelength:.1f} PPW)"
    )
    print(f"  Duration:   {config.duration * 1e6:.1f} us ({config.num_time_steps} steps)")
    print(
        f"  Sensor:     ({config.sensor_position[0] * 1e3:.1f}, {config.sensor_position[1] * 1e3:.1f}, {config.sensor_position[2] * 1e3:.1f}) mm"
    )
    print()

    return config


def select_simulators() -> list:
    """
    Select available simulators.

    Returns:
        List of SimulatorType to run
    """
    simulators = []

    # Check if pykwavers-only mode is requested
    pykwavers_only = os.environ.get("KWAVERS_PYKWAVERS_ONLY", "").lower() in ("1", "true", "yes")

    # Always include pykwavers solvers if available
    if PYKWAVERS_AVAILABLE:
        simulators.extend(
            [
                SimulatorType.PYKWAVERS_FDTD,
                SimulatorType.PYKWAVERS_PSTD,
                SimulatorType.PYKWAVERS_HYBRID,
            ]
        )

    # Add k-wave-python if available and not in pykwavers-only mode
    if KWAVE_PYTHON_AVAILABLE and not pykwavers_only:
        simulators.append(SimulatorType.KWAVE_PYTHON)

    # Add k-Wave MATLAB if available and not in pykwavers-only mode
    if MATLAB_AVAILABLE and not pykwavers_only:
        simulators.append(SimulatorType.KWAVE_MATLAB)

    if pykwavers_only:
        print("Running in pykwavers-only mode (KWAVERS_PYKWAVERS_ONLY set)")
        print()

    print(f"Selected {len(simulators)} simulator(s):")
    for sim in simulators:
        print(f"  - {sim.value}")
    print()

    return simulators


def determine_reference(simulators: list) -> SimulatorType:
    """
    Determine reference simulator for error computation.

    Priority: k-wave-python > k-Wave MATLAB > pykwavers PSTD

    Args:
        simulators: List of available simulators

    Returns:
        Reference SimulatorType
    """
    if SimulatorType.KWAVE_PYTHON in simulators:
        return SimulatorType.KWAVE_PYTHON
    elif SimulatorType.KWAVE_MATLAB in simulators:
        return SimulatorType.KWAVE_MATLAB
    elif SimulatorType.PYKWAVERS_PSTD in simulators:
        return SimulatorType.PYKWAVERS_PSTD
    else:
        return simulators[0]


def save_results(comparison: ComparisonResult, output_dir: Path):
    """
    Save comparison results to disk.

    Args:
        comparison: ComparisonResult object
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    plot_comparison(comparison, output_path=output_dir / "comparison.png")

    # Save metrics as CSV
    try:
        import pandas as pd

        metrics_data = []
        for sim_type, result in comparison.results.items():
            row = {
                "simulator": sim_type.value,
                "execution_time": result.execution_time,
                "memory_usage": result.memory_usage if result.memory_usage else 0,
            }

            # Add error metrics if not reference
            if sim_type != comparison.reference:
                metrics = comparison.error_metrics[sim_type]
                row.update(
                    {
                        "l2_error": metrics["l2_error"],
                        "linf_error": metrics["linf_error"],
                        "rmse": metrics["rmse"],
                        "max_abs_error": metrics["max_abs_error"],
                        "correlation": metrics["correlation"],
                        "validation_passed": comparison.validation_passed[sim_type],
                    }
                )
            else:
                row.update(
                    {
                        "l2_error": 0.0,
                        "linf_error": 0.0,
                        "rmse": 0.0,
                        "max_abs_error": 0.0,
                        "correlation": 1.0,
                        "validation_passed": True,
                    }
                )

            metrics_data.append(row)

        df = pd.DataFrame(metrics_data)
        csv_path = output_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

    except ImportError:
        print("pandas not available, skipping CSV export")

    # Save validation report
    report_path = output_dir / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write(comparison.validation_report)
    print(f"Validation report saved to {report_path}")

    # Save raw sensor data
    data_path = output_dir / "sensor_data.npz"
    data_dict = {}
    for sim_type, result in comparison.results.items():
        prefix = sim_type.value.replace("_", "")
        data_dict[f"{prefix}_pressure"] = result.pressure
        data_dict[f"{prefix}_time"] = result.time
        data_dict[f"{prefix}_execution_time"] = result.execution_time

    np.savez_compressed(data_path, **data_dict)
    print(f"Sensor data saved to {data_path}")

    print()
    print(f"All results saved to {output_dir.absolute()}")


def main():
    """Main comparison workflow."""
    print_header()

    # Check availability
    if not check_availability():
        return 1

    # Create configuration
    config = create_config()

    # Select simulators
    simulators = select_simulators()
    if not simulators:
        print("ERROR: No simulators available")
        return 1

    # Determine reference
    reference = determine_reference(simulators)
    print(f"Reference simulator: {reference.value}")
    print()

    # Run comparison
    try:
        comparison = run_comparison(config, simulators, reference=reference)

        # Print validation report
        print()
        print(comparison.validation_report)

        # Save results
        output_dir = Path(__file__).parent / "results"
        save_results(comparison, output_dir)

        # Summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Performance ranking
        print("\nPerformance Ranking (fastest to slowest):")
        sorted_results = sorted(comparison.results.items(), key=lambda x: x[1].execution_time)
        for i, (sim_type, result) in enumerate(sorted_results, 1):
            speedup = comparison.results[reference].execution_time / result.execution_time
            print(
                f"{i}. {sim_type.value:25s} {result.execution_time:8.3f}s  ({speedup:5.2f}x vs reference)"
            )

        # Accuracy ranking (if errors computed)
        if comparison.error_metrics:
            print("\nAccuracy Ranking (most accurate to least):")
            sorted_errors = sorted(comparison.error_metrics.items(), key=lambda x: x[1]["l2_error"])
            for i, (sim_type, metrics) in enumerate(sorted_errors, 1):
                status = "[OK]" if comparison.validation_passed[sim_type] else "[FAIL]"
                print(
                    f"{i}. {sim_type.value:25s} L2={metrics['l2_error']:.2e}, L∞={metrics['linf_error']:.2e} {status}"
                )

        # Overall pass/fail
        print()
        all_passed = all(comparison.validation_passed.values())
        if all_passed:
            print("[OK] ALL SIMULATORS PASSED VALIDATION")
        else:
            failed = [
                sim.value for sim, passed in comparison.validation_passed.items() if not passed
            ]
            print(f"[X] VALIDATION FAILED FOR: {', '.join(failed)}")

        print("=" * 80)

        return 0 if all_passed else 1

    except Exception as e:
        print(f"\n[X] Comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
