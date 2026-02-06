#!/usr/bin/env python3
"""
Minimal diagnostic to debug source amplitude issues.

This script creates a simple plane wave source and checks:
1. Input signal amplitude
2. Mask values
3. Output pressure amplitude
4. Comparison with k-wave-python

Author: Ryan Clanton
Date: 2026-02-04
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "python"))

import pykwavers as kw
from pykwavers.kwave_python_bridge import (
    GridParams,
    KWavePythonBridge,
    MediumParams,
    SensorParams,
    SourceParams,
)

print("=" * 80)
print("Source Amplitude Diagnostic")
print("=" * 80)
print()

# Configuration
grid_size = 64  # Increased to accommodate PML
spacing = 0.1e-3  # 0.1 mm
c0 = 1500.0  # m/s
rho0 = 1000.0  # kg/m³
freq = 1e6  # 1 MHz
amplitude = 1e5  # 100 kPa
duration = 5e-6  # 5 us
cfl = 0.3
pml_size = 10  # PML thickness

# Compute time parameters
dx = spacing
dt = cfl * dx / c0
nt = int(duration / dt)

print(f"Grid: {grid_size}³")
print(f"Spacing: {spacing * 1e3:.2f} mm")
print(f"Time step: {dt * 1e9:.2f} ns")
print(f"Steps: {nt}")
print(f"Source: {freq * 1e-6:.1f} MHz, {amplitude * 1e-3:.0f} kPa")
print()

# ============================================================================
# Test 1: pykwavers FDTD
# ============================================================================
print("-" * 80)
print("Test 1: pykwavers FDTD")
print("-" * 80)

# Create grid
grid = kw.Grid(nx=grid_size, ny=grid_size, nz=grid_size, dx=spacing, dy=spacing, dz=spacing)

# Medium
medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

# Source: plane wave at z=0
mask = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
mask[:, :, 0] = 1.0

# Time signal
t = np.arange(nt) * dt
signal = amplitude * np.sin(2 * np.pi * freq * t)

print(f"Input signal:")
print(f"  Shape: {signal.shape}")
print(f"  Min: {signal.min():.3e} Pa")
print(f"  Max: {signal.max():.3e} Pa")
print(f"  Mean: {signal.mean():.3e} Pa")
print(f"  Expected amplitude: {amplitude:.3e} Pa")
print()

print(f"Mask:")
print(f"  Shape: {mask.shape}")
print(f"  Active points: {np.sum(mask > 0)}")
print(f"  Mask value at (0,0,0): {mask[0, 0, 0]}")
print()

source = kw.Source.from_mask(mask, signal, frequency=freq)

# Sensor at center
sensor_pos = (grid_size // 2 * spacing, grid_size // 2 * spacing, grid_size // 2 * spacing)
sensor = kw.Sensor.point(position=sensor_pos)

# Run FDTD
sim_fdtd = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.FDTD)
result_fdtd = sim_fdtd.run(time_steps=nt, dt=dt)

print(f"FDTD Output:")
print(f"  Pressure shape: {result_fdtd.sensor_data.shape}")
print(f"  Min: {result_fdtd.sensor_data.min():.3e} Pa")
print(f"  Max: {result_fdtd.sensor_data.max():.3e} Pa")
print(f"  Mean: {result_fdtd.sensor_data.mean():.3e} Pa")
print(f"  Std: {result_fdtd.sensor_data.std():.3e} Pa")
print()

# ============================================================================
# Test 2: pykwavers PSTD
# ============================================================================
print("-" * 80)
print("Test 2: pykwavers PSTD")
print("-" * 80)

sim_pstd = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
result_pstd = sim_pstd.run(time_steps=nt, dt=dt)

print(f"PSTD Output:")
print(f"  Pressure shape: {result_pstd.sensor_data.shape}")
print(f"  Min: {result_pstd.sensor_data.min():.3e} Pa")
print(f"  Max: {result_pstd.sensor_data.max():.3e} Pa")
print(f"  Mean: {result_pstd.sensor_data.mean():.3e} Pa")
print(f"  Std: {result_pstd.sensor_data.std():.3e} Pa")
print()

# ============================================================================
# Test 3: k-wave-python
# ============================================================================
print("-" * 80)
print("Test 3: k-wave-python")
print("-" * 80)

# Grid
grid_kw = GridParams(
    Nx=grid_size,
    Ny=grid_size,
    Nz=grid_size,
    dx=spacing,
    dy=spacing,
    dz=spacing,
    dt=dt,
    pml_size=pml_size,
)

# Medium
medium_kw = MediumParams(sound_speed=c0, density=rho0)

# Source
p_mask = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
p_mask[:, :, 0] = True
p_signal = amplitude * np.sin(2 * np.pi * freq * t)

source_kw = SourceParams(p_mask=p_mask, p=p_signal, frequency=freq, amplitude=amplitude)

# Sensor
sensor_mask = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
ix = grid_size // 2
iy = grid_size // 2
iz = grid_size // 2
sensor_mask[ix, iy, iz] = True

sensor_kw = SensorParams(mask=sensor_mask, record=["p"])

# Run
bridge = KWavePythonBridge(cache_dir="./kwave_cache")
result_kw = bridge.run_simulation(grid_kw, medium_kw, source_kw, sensor_kw, nt)

print(f"k-wave-python Output:")
print(f"  Pressure shape: {result_kw.sensor_data.shape}")
print(f"  Min: {result_kw.sensor_data.min():.3e} Pa")
print(f"  Max: {result_kw.sensor_data.max():.3e} Pa")
print(f"  Mean: {result_kw.sensor_data.mean():.3e} Pa")
print(f"  Std: {result_kw.sensor_data.std():.3e} Pa")
print()

# ============================================================================
# Comparison
# ============================================================================
print("=" * 80)
print("AMPLITUDE COMPARISON")
print("=" * 80)
print()

fdtd_max = np.abs(result_fdtd.sensor_data).max()
pstd_max = np.abs(result_pstd.sensor_data).max()
kw_max = np.abs(result_kw.sensor_data).max()

print(f"Expected amplitude:    {amplitude:.3e} Pa ({amplitude * 1e-3:.1f} kPa)")
print()
print(f"FDTD max amplitude:    {fdtd_max:.3e} Pa ({fdtd_max * 1e-3:.1f} kPa)")
print(f"  Ratio vs expected:   {fdtd_max / amplitude:.2f}x")
print(f"  Ratio vs k-wave:     {fdtd_max / kw_max:.2f}x")
print()
print(f"PSTD max amplitude:    {pstd_max:.3e} Pa ({pstd_max * 1e-3:.1f} kPa)")
print(f"  Ratio vs expected:   {pstd_max / amplitude:.2f}x")
print(f"  Ratio vs k-wave:     {pstd_max / kw_max:.2f}x")
print()
print(f"k-wave max amplitude:  {kw_max:.3e} Pa ({kw_max * 1e-3:.1f} kPa)")
print(f"  Ratio vs expected:   {kw_max / amplitude:.2f}x")
print()

# Check for issues
issues = []
if fdtd_max / kw_max > 2.0 or fdtd_max / kw_max < 0.5:
    issues.append(f"FDTD amplitude {fdtd_max / kw_max:.1f}x different from k-wave")
if pstd_max / kw_max > 2.0 or pstd_max / kw_max < 0.5:
    issues.append(f"PSTD amplitude {pstd_max / kw_max:.1f}x different from k-wave")

if issues:
    print("ISSUES DETECTED:")
    for issue in issues:
        print(f"  [X] {issue}")
else:
    print("[OK] All amplitudes within 2x of k-wave-python")
print()

# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Time series
ax = axes[0]
t_us = result_fdtd.time * 1e6
ax.plot(t_us, result_fdtd.sensor_data.flatten(), label="FDTD", alpha=0.7)
ax.plot(t_us, result_pstd.sensor_data.flatten(), label="PSTD", alpha=0.7)
ax.plot(
    result_kw.time_array * 1e6,
    result_kw.sensor_data.flatten(),
    label="k-wave-python",
    alpha=0.7,
    linestyle="--",
)
ax.set_xlabel("Time (μs)")
ax.set_ylabel("Pressure (Pa)")
ax.set_title("Sensor Time Series")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: First few cycles (zoomed)
ax = axes[1]
zoom_samples = min(200, len(t_us))
ax.plot(
    t_us[:zoom_samples],
    result_fdtd.sensor_data.flatten()[:zoom_samples],
    label="FDTD",
    marker=".",
    markersize=2,
)
ax.plot(
    t_us[:zoom_samples],
    result_pstd.sensor_data.flatten()[:zoom_samples],
    label="PSTD",
    marker=".",
    markersize=2,
)
ax.plot(
    result_kw.time_array[:zoom_samples] * 1e6,
    result_kw.sensor_data.flatten()[:zoom_samples],
    label="k-wave-python",
    marker=".",
    markersize=2,
    linestyle="--",
)
ax.set_xlabel("Time (μs)")
ax.set_ylabel("Pressure (Pa)")
ax.set_title("First Few Cycles (Zoomed)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Amplitude comparison
ax = axes[2]
simulators = ["Expected", "FDTD", "PSTD", "k-wave"]
amplitudes = [amplitude, fdtd_max, pstd_max, kw_max]
colors = ["gray", "blue", "green", "red"]
bars = ax.bar(simulators, [a * 1e-3 for a in amplitudes], color=colors, alpha=0.7)
ax.set_ylabel("Max Amplitude (kPa)")
ax.set_title("Amplitude Comparison")
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, amp in zip(bars, amplitudes):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{amp * 1e-3:.1f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig("debug_amplitude_comparison.png", dpi=150)
print(f"Plot saved: debug_amplitude_comparison.png")
print()

print("=" * 80)
print("Diagnostic complete")
print("=" * 80)
