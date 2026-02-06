#!/usr/bin/env python3
"""
Validation Analysis: pykwavers vs k-wave-python

This script performs detailed analysis of differences between pykwavers and k-wave-python
to identify sources of discrepancy and validate implementations.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))

from pykwavers.comparison import (
    SimulationConfig,
    SimulatorType,
    compute_error_metrics,
    run_kwave_python,
    run_pykwavers,
)


def analyze_plane_wave_propagation():
    """Analyze plane wave propagation differences."""
    print("=" * 80)
    print("Plane Wave Propagation Analysis")
    print("=" * 80)

    # Test configuration
    config = SimulationConfig(
        grid_shape=(64, 64, 64),
        grid_spacing=(0.1e-3, 0.1e-3, 0.1e-3),
        sound_speed=1500.0,
        density=1000.0,
        source_frequency=1e6,
        source_amplitude=1e5,
        duration=10e-6,
        source_position=None,  # Plane wave
        sensor_position=(3.2e-3, 3.2e-3, 3.2e-3),
        pml_size=10,
    )

    print(f"Grid: {config.grid_shape}")
    print(f"Spacing: {config.grid_spacing[0] * 1e3:.2f} mm")
    print(f"Frequency: {config.source_frequency * 1e-6:.1f} MHz")
    print(f"Wavelength: {config.wavelength * 1e3:.2f} mm")
    print(f"Points per wavelength: {config.points_per_wavelength:.1f}")
    print(f"Time steps: {config.num_time_steps}")
    print()

    # Run both simulators
    print("Running pykwavers (FDTD)...")
    result_pyfdtd = run_pykwavers(config, solver_type="fdtd")
    print(f"  Complete in {result_pyfdtd.execution_time:.2f}s")

    print("Running pykwavers (PSTD)...")
    result_pypstd = run_pykwavers(config, solver_type="pstd")
    print(f"  Complete in {result_pypstd.execution_time:.2f}s")

    print("Running k-wave-python...")
    result_kwave = run_kwave_python(config)
    print(f"  Complete in {result_kwave.execution_time:.2f}s")

    # Compute metrics
    print("\nError Metrics:")
    print("-" * 80)

    metrics_fdtd_vs_kwave = compute_error_metrics(result_kwave.pressure, result_pyfdtd.pressure)
    print("\npykwavers (FDTD) vs k-wave-python:")
    for key, value in metrics_fdtd_vs_kwave.items():
        print(f"  {key}: {value:.4e}")

    metrics_pstd_vs_kwave = compute_error_metrics(result_kwave.pressure, result_pypstd.pressure)
    print("\npykwavers (PSTD) vs k-wave-python:")
    for key, value in metrics_pstd_vs_kwave.items():
        print(f"  {key}: {value:.4e}")

    metrics_fdtd_vs_pstd = compute_error_metrics(result_pypstd.pressure, result_pyfdtd.pressure)
    print("\npykwavers (FDTD) vs pykwavers (PSTD):")
    for key, value in metrics_fdtd_vs_pstd.items():
        print(f"  {key}: {value:.4e}")

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Time series
    ax1 = axes[0]
    ax1.plot(
        result_kwave.time * 1e6,
        result_kwave.pressure / 1e3,
        "k-",
        label="k-wave-python",
        linewidth=2,
    )
    ax1.plot(
        result_pyfdtd.time * 1e6,
        result_pyfdtd.pressure / 1e3,
        "r--",
        label="pykwavers (FDTD)",
        linewidth=1.5,
        alpha=0.8,
    )
    ax1.plot(
        result_pypstd.time * 1e6,
        result_pypstd.pressure / 1e3,
        "b:",
        label="pykwavers (PSTD)",
        linewidth=1.5,
        alpha=0.8,
    )
    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("Pressure [kPa]")
    ax1.set_title("Pressure Time Series Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Errors
    ax2 = axes[1]
    min_len = min(len(result_kwave.pressure), len(result_pyfdtd.pressure))
    error_fdtd = (result_pyfdtd.pressure[:min_len] - result_kwave.pressure[:min_len]) / 1e3
    min_len_pstd = min(len(result_kwave.pressure), len(result_pypstd.pressure))
    error_pstd = (
        result_pypstd.pressure[:min_len_pstd] - result_kwave.pressure[:min_len_pstd]
    ) / 1e3

    ax2.plot(
        result_kwave.time[:min_len] * 1e6,
        error_fdtd,
        "r-",
        label="FDTD vs k-wave",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.plot(
        result_kwave.time[:min_len_pstd] * 1e6,
        error_pstd,
        "b-",
        label="PSTD vs k-wave",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.set_xlabel("Time [us]")
    ax2.set_ylabel("Pressure Error [kPa]")
    ax2.set_title("Error vs k-wave-python")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)

    # Spectral analysis
    ax3 = axes[2]
    from numpy.fft import fft, fftfreq

    # FFT of signals
    dt = result_kwave.time[1] - result_kwave.time[0]
    freqs = fftfreq(len(result_kwave.pressure), dt)

    fft_kwave = np.abs(fft(result_kwave.pressure))
    fft_fdtd = np.abs(fft(result_pyfdtd.pressure[: len(result_kwave.pressure)]))
    fft_pstd = np.abs(fft(result_pypstd.pressure[: len(result_kwave.pressure)]))

    # Plot positive frequencies only
    pos_mask = freqs > 0
    ax3.semilogy(
        freqs[pos_mask] * 1e-6, fft_kwave[pos_mask], "k-", label="k-wave-python", linewidth=2
    )
    ax3.semilogy(
        freqs[pos_mask] * 1e-6,
        fft_fdtd[pos_mask],
        "r--",
        label="pykwavers (FDTD)",
        linewidth=1.5,
        alpha=0.8,
    )
    ax3.semilogy(
        freqs[pos_mask] * 1e-6,
        fft_pstd[pos_mask],
        "b:",
        label="pykwavers (PSTD)",
        linewidth=1.5,
        alpha=0.8,
    )
    ax3.axvline(
        config.source_frequency * 1e-6,
        color="g",
        linestyle="--",
        label="Source frequency",
        alpha=0.5,
    )
    ax3.set_xlabel("Frequency [MHz]")
    ax3.set_ylabel("FFT Magnitude")
    ax3.set_title("Frequency Content")
    ax3.set_xlim([0, 5])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("validation_analysis.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: validation_analysis.png")

    return {
        "fdtd_vs_kwave": metrics_fdtd_vs_kwave,
        "pstd_vs_kwave": metrics_pstd_vs_kwave,
        "fdtd_vs_pstd": metrics_fdtd_vs_pstd,
    }


def main():
    """Main entry point."""
    print("=" * 80)
    print("pykwavers vs k-wave-python Validation Analysis")
    print("=" * 80)
    print()

    # Check availability
    try:
        import pykwavers as kw

        print("pykwavers: [OK]")
    except ImportError:
        print("pykwavers: [X] Not available")
        return

    try:
        from kwave.kgrid import kWaveGrid

        print("k-wave-python: [OK]")
    except ImportError:
        print("k-wave-python: [X] Not available")
        return

    print()

    # Run analysis
    try:
        results = analyze_plane_wave_propagation()

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        # Check if PSTD meets criteria
        pstd_metrics = results["pstd_vs_kwave"]
        l2_pass = pstd_metrics["l2_error"] < 0.01
        linf_pass = pstd_metrics["linf_error"] < 0.05
        corr_pass = pstd_metrics["correlation"] > 0.95

        print(f"\npykwavers PSTD vs k-wave-python:")
        print(f"  L2 error: {pstd_metrics['l2_error']:.4e} {'[OK]' if l2_pass else '[X]'} (< 0.01)")
        print(
            f"  Linf error: {pstd_metrics['linf_error']:.4e} {'[OK]' if linf_pass else '[X]'} (< 0.05)"
        )
        print(
            f"  Correlation: {pstd_metrics['correlation']:.4f} {'[OK]' if corr_pass else '[X]'} (> 0.95)"
        )
        print(f"  Overall: {'[OK] PASS' if (l2_pass and linf_pass and corr_pass) else '[X] FAIL'}")

    except Exception as e:
        print(f"[X] Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
