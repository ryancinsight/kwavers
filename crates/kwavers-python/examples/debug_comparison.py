#!/usr/bin/env python3
"""
Debug Comparison: Detailed diagnostics for pykwavers vs k-wave-python

This script performs detailed analysis of the differences between
pykwavers and k-wave-python to identify the root cause of discrepancies.

Author: Ryan Clanton (@ryancinsight)
Date: 2025-01-20
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import pykwavers as kw
from pykwavers.comparison import SimulationConfig, config_to_kwave_python, config_to_pykwavers
from pykwavers.kwave_python_bridge import KWavePythonBridge


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_diagnostic_plane_wave():
    """Run diagnostic for plane wave source."""
    print_section("DIAGNOSTIC: PLANE WAVE SOURCE")

    # Create simple test configuration
    config = SimulationConfig(
        grid_shape=(64, 64, 64),
        grid_spacing=(0.1e-3, 0.1e-3, 0.1e-3),
        sound_speed=1500.0,
        density=1000.0,
        source_frequency=1e6,
        source_amplitude=1e5,
        source_position=None,  # Plane wave
        sensor_position=(3.2e-3, 3.2e-3, 3.2e-3),
        duration=10e-6,
        pml_size=10,
        cfl=0.3,
    )

    return run_comparison_test(config, "plane_wave")


def run_diagnostic_point_source():
    """Run diagnostic for point source."""
    print_section("DIAGNOSTIC: POINT SOURCE")

    # Create test configuration with point source
    config = SimulationConfig(
        grid_shape=(64, 64, 64),
        grid_spacing=(0.1e-3, 0.1e-3, 0.1e-3),
        sound_speed=1500.0,
        density=1000.0,
        source_frequency=1e6,
        source_amplitude=1e5,
        source_position=(0.32e-3, 0.32e-3, 0.32e-3),  # Point source near corner
        sensor_position=(3.2e-3, 3.2e-3, 3.2e-3),
        duration=10e-6,
        pml_size=10,
        cfl=0.3,
    )

    return run_comparison_test(config, "point_source")


def run_comparison_test(config, test_name):
    """Run comparison test for given configuration."""

    print("\nConfiguration:")
    print(f"  Grid: {config.grid_shape}")
    print(f"  Spacing: {config.grid_spacing}")
    print(f"  Sensor position (m): {config.sensor_position}")
    print(
        f"  Sensor index: ({int(config.sensor_position[0] / config.grid_spacing[0])}, "
        f"{int(config.sensor_position[1] / config.grid_spacing[1])}, "
        f"{int(config.sensor_position[2] / config.grid_spacing[2])})"
    )
    print(f"  Duration: {config.duration * 1e6:.1f} us")
    print(f"  Time steps: {config.num_time_steps}")

    # Run pykwavers PSTD
    print_section("Running pykwavers PSTD")
    grid_kw, medium_kw, source_kw, sensor_kw, nt_kw, dt_kw = config_to_pykwavers(config)

    print(f"Source type: {source_kw.source_type}")
    print(f"Frequency: {source_kw.frequency:.2e} Hz")
    print(f"Amplitude: {source_kw.amplitude:.2e} Pa")
    print(f"Time steps: {nt_kw}")
    print(f"dt: {dt_kw}")

    sim_kw = kw.Simulation(grid_kw, medium_kw, source_kw, sensor_kw, solver=kw.SolverType.PSTD)
    result_kw = sim_kw.run(time_steps=nt_kw, dt=dt_kw)

    p_kw = result_kw.sensor_data.flatten()
    t_kw = result_kw.time

    print(f"Result shape: {p_kw.shape}")
    print(f"Time array shape: {t_kw.shape}")
    print(f"Time range: {t_kw[0]:.2e} to {t_kw[-1]:.2e} s")
    print(f"Pressure min: {np.min(p_kw):.2e} Pa")
    print(f"Pressure max: {np.max(p_kw):.2e} Pa")
    print(f"Pressure mean: {np.mean(p_kw):.2e} Pa")
    print(f"Pressure std: {np.std(p_kw):.2e} Pa")

    # Run k-wave-python
    print_section("Running k-wave-python")
    grid_kwave, medium_kwave, source_kwave, sensor_kwave, nt_kwave = config_to_kwave_python(config)

    print(f"Grid: {grid_kwave.shape}")
    print(f"Source mask shape: {source_kwave.p_mask.shape}")
    print(f"Source mask sum: {np.sum(source_kwave.p_mask)}")
    print(f"Source signal shape: {source_kwave.p.shape}")
    print(f"Sensor mask shape: {sensor_kwave.mask.shape}")
    print(f"Sensor mask sum: {np.sum(sensor_kwave.mask)}")
    print(f"Time steps: {nt_kwave}")

    bridge = KWavePythonBridge()
    result_kwave = bridge.run_simulation(
        grid_kwave, medium_kwave, source_kwave, sensor_kwave, nt_kwave
    )

    p_kwave = result_kwave.sensor_data.flatten()
    t_kwave = result_kwave.time_array

    print(f"Result shape: {p_kwave.shape}")
    print(f"Time array shape: {t_kwave.shape}")
    print(f"Time range: {t_kwave[0]:.2e} to {t_kwave[-1]:.2e} s")
    print(f"Pressure min: {np.min(p_kwave):.2e} Pa")
    print(f"Pressure max: {np.max(p_kwave):.2e} Pa")
    print(f"Pressure mean: {np.mean(p_kwave):.2e} Pa")
    print(f"Pressure std: {np.std(p_kwave):.2e} Pa")

    # Align time arrays (k-wave may have different length)
    print_section("Time Array Alignment")
    print(f"pykwavers length: {len(t_kw)}")
    print(f"k-wave-python length: {len(t_kwave)}")

    min_len = min(len(p_kw), len(p_kwave))
    p_kw_aligned = p_kw[:min_len]
    p_kwave_aligned = p_kwave[:min_len]
    t_aligned = t_kw[:min_len]

    print(f"Aligned length: {min_len}")

    # Compute error metrics
    print_section("Error Analysis")

    # Basic statistics
    diff = p_kw_aligned - p_kwave_aligned
    print(f"\nDifference statistics:")
    print(f"  Mean difference: {np.mean(diff):.2e} Pa")
    print(f"  Std difference: {np.std(diff):.2e} Pa")
    print(f"  Min difference: {np.min(diff):.2e} Pa")
    print(f"  Max difference: {np.max(diff):.2e} Pa")

    # Normalized errors
    p_kwave_rms = np.sqrt(np.mean(p_kwave_aligned**2))
    l2_error = np.linalg.norm(diff) / np.linalg.norm(p_kwave_aligned)
    linf_error = np.max(np.abs(diff)) / np.max(np.abs(p_kwave_aligned))
    rmse = np.sqrt(np.mean(diff**2))
    rel_rmse = rmse / p_kwave_rms

    print(f"\nNormalized errors:")
    print(f"  L2 error: {l2_error:.2e}")
    print(f"  Linf error: {linf_error:.2e}")
    print(f"  RMSE: {rmse:.2e} Pa")
    print(f"  Relative RMSE: {rel_rmse:.2e}")

    # Correlation
    if np.std(p_kw_aligned) > 0 and np.std(p_kwave_aligned) > 0:
        correlation = np.corrcoef(p_kw_aligned, p_kwave_aligned)[0, 1]
        print(f"  Correlation: {correlation:.4f}")
    else:
        print(f"  Correlation: N/A (zero variance)")

    # Phase analysis
    print(f"\nPhase analysis:")
    # Find first significant arrival (threshold: 1% of max)
    threshold_kw = 0.01 * np.max(np.abs(p_kw_aligned))
    threshold_kwave = 0.01 * np.max(np.abs(p_kwave_aligned))

    arrival_idx_kw = np.where(np.abs(p_kw_aligned) > threshold_kw)[0]
    arrival_idx_kwave = np.where(np.abs(p_kwave_aligned) > threshold_kwave)[0]

    if len(arrival_idx_kw) > 0 and len(arrival_idx_kwave) > 0:
        arrival_kw = arrival_idx_kw[0]
        arrival_kwave = arrival_idx_kwave[0]
        print(
            f"  First arrival (pykwavers): index {arrival_kw}, t={t_aligned[arrival_kw] * 1e6:.3f} us"
        )
        print(
            f"  First arrival (k-wave): index {arrival_kwave}, t={t_aligned[arrival_kwave] * 1e6:.3f} us"
        )
        print(
            f"  Arrival time difference: {(arrival_kw - arrival_kwave) * t_aligned[1] * 1e6:.3f} us"
        )
    else:
        print("  Could not detect arrivals")

    # Frequency analysis
    print(f"\nFrequency analysis:")
    # FFT of signals
    fft_kw = np.fft.rfft(p_kw_aligned)
    fft_kwave = np.fft.rfft(p_kwave_aligned)
    freqs = np.fft.rfftfreq(len(p_kw_aligned), t_aligned[1] - t_aligned[0])

    # Find peak frequency
    peak_idx_kw = np.argmax(np.abs(fft_kw))
    peak_idx_kwave = np.argmax(np.abs(fft_kwave))

    print(f"  Peak frequency (pykwavers): {freqs[peak_idx_kw] * 1e-6:.3f} MHz")
    print(f"  Peak frequency (k-wave): {freqs[peak_idx_kwave] * 1e-6:.3f} MHz")
    print(f"  Expected frequency: {config.source_frequency * 1e-6:.3f} MHz")

    # Create diagnostic plots
    print_section("Generating Diagnostic Plots")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: Full time series
    ax = axes[0, 0]
    ax.plot(t_aligned * 1e6, p_kw_aligned * 1e-3, "b-", label="pykwavers", alpha=0.7)
    ax.plot(t_aligned * 1e6, p_kwave_aligned * 1e-3, "r--", label="k-wave-python", alpha=0.7)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title("Full Time Series")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoomed view (first 20% of time)
    ax = axes[0, 1]
    zoom_len = len(t_aligned) // 5
    ax.plot(t_aligned[:zoom_len] * 1e6, p_kw_aligned[:zoom_len] * 1e-3, "b-", label="pykwavers")
    ax.plot(
        t_aligned[:zoom_len] * 1e6, p_kwave_aligned[:zoom_len] * 1e-3, "r--", label="k-wave-python"
    )
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title("Zoomed View (First 20%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Difference
    ax = axes[1, 0]
    ax.plot(t_aligned * 1e6, diff * 1e-3, "k-", linewidth=0.5)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Difference [kPa]")
    ax.set_title("Difference (pykwavers - k-wave)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Relative error
    ax = axes[1, 1]
    rel_error = np.abs(diff) / (np.abs(p_kwave_aligned) + 1e-10)
    ax.semilogy(t_aligned * 1e6, rel_error, "k-", linewidth=0.5)
    ax.set_xlabel("Time [us]")
    ax.set_ylabel("Relative Error")
    ax.set_title("Relative Error (log scale)")
    ax.grid(True, alpha=0.3)

    # Plot 5: Frequency spectrum
    ax = axes[2, 0]
    ax.semilogy(freqs * 1e-6, np.abs(fft_kw), "b-", label="pykwavers", alpha=0.7)
    ax.semilogy(freqs * 1e-6, np.abs(fft_kwave), "r--", label="k-wave-python", alpha=0.7)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Spectrum")
    ax.set_xlim([0, 3 * config.source_frequency * 1e-6])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Cross-correlation
    ax = axes[2, 1]
    # Compute cross-correlation
    corr = np.correlate(
        p_kw_aligned - np.mean(p_kw_aligned),
        p_kwave_aligned - np.mean(p_kwave_aligned),
        mode="full",
    )
    corr = corr / (np.std(p_kw_aligned) * np.std(p_kwave_aligned) * len(p_kw_aligned))
    lags = np.arange(-len(p_kw_aligned) + 1, len(p_kw_aligned))
    center = len(p_kw_aligned) - 1
    window = 100
    ax.plot(
        lags[center - window : center + window] * (t_aligned[1] - t_aligned[0]) * 1e6,
        corr[center - window : center + window],
        "k-",
    )
    ax.set_xlabel("Lag [us]")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-correlation")
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(__file__).parent / "results" / f"debug_comparison_{test_name}.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Diagnostic plot saved: {output_path}")
    plt.close()

    # Save raw data
    print_section("Saving Raw Data")
    data_path = Path(__file__).parent / "results" / f"debug_data_{test_name}.npz"
    np.savez_compressed(
        data_path,
        t_kw=t_aligned,
        p_kw=p_kw_aligned,
        p_kwave=p_kwave_aligned,
        diff=diff,
        config_grid=config.grid_shape,
        config_spacing=config.grid_spacing,
        config_sensor=config.sensor_position,
    )
    print(f"Raw data saved: {data_path}")

    print_section("SUMMARY")
    print(f"\nKey findings:")
    print(f"  1. L2 error: {l2_error:.2e} ({'PASS' if l2_error < 0.01 else 'FAIL'})")
    print(f"  2. Correlation: {correlation:.4f} ({'PASS' if correlation > 0.99 else 'FAIL'})")
    print(
        f"  3. Peak frequency match: {np.abs(freqs[peak_idx_kw] - freqs[peak_idx_kwave]) * 1e-6:.3f} MHz"
    )
    print(
        f"  4. Signal magnitude ratio: {np.max(np.abs(p_kw_aligned)) / np.max(np.abs(p_kwave_aligned)):.3f}"
    )

    if correlation < 0.5:
        print(
            "\nWARNING: Very low correlation suggests fundamental difference in simulation setup!"
        )
        print("Possible causes:")
        print("  - Source position/orientation mismatch")
        print("  - Sensor position mismatch")
        print("  - Time stepping/synchronization issues")
        print("  - Boundary condition differences")

    return {
        "l2_error": l2_error,
        "correlation": correlation,
        "magnitude_ratio": np.max(np.abs(p_kw_aligned)) / np.max(np.abs(p_kwave_aligned)),
    }


def run_diagnostic():
    """Run all diagnostic tests."""
    print_section("DIAGNOSTIC COMPARISON: pykwavers vs k-wave-python")

    # Test 1: Point source (simpler, should work better)
    print("\n" + "=" * 80)
    print("TEST 1: POINT SOURCE")
    print("=" * 80)
    results_point = run_diagnostic_point_source()

    # Test 2: Plane wave (current problem)
    print("\n" + "=" * 80)
    print("TEST 2: PLANE WAVE")
    print("=" * 80)
    results_plane = run_diagnostic_plane_wave()

    # Summary
    print_section("OVERALL SUMMARY")
    print("\nPoint Source:")
    print(f"  L2 error: {results_point['l2_error']:.2e}")
    print(f"  Correlation: {results_point['correlation']:.4f}")
    print(f"  Magnitude ratio: {results_point['magnitude_ratio']:.3f}")

    print("\nPlane Wave:")
    print(f"  L2 error: {results_plane['l2_error']:.2e}")
    print(f"  Correlation: {results_plane['correlation']:.4f}")
    print(f"  Magnitude ratio: {results_plane['magnitude_ratio']:.3f}")

    if results_point["correlation"] > 0.9 and results_plane["correlation"] < 0.5:
        print("\nCONCLUSION: Point source works, plane wave does not.")
        print("Issue is likely in plane wave source implementation.")
    elif results_point["correlation"] < 0.5:
        print("\nCONCLUSION: Both sources fail validation.")
        print("Issue is likely fundamental (time stepping, boundary conditions, etc.)")
    else:
        print("\nCONCLUSION: Both sources work reasonably well.")

    return 0 if results_point["correlation"] > 0.9 else 1


if __name__ == "__main__":
    sys.exit(run_diagnostic())
