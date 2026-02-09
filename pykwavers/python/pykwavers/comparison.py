#!/usr/bin/env python3
"""
Unified Comparison Framework for pykwavers, k-Wave MATLAB, and k-wave-python

This module provides comprehensive comparison and validation utilities for acoustic
simulations across three implementations:
1. pykwavers: Rust-backed FDTD/PSTD/Hybrid solvers
2. k-Wave (MATLAB): Reference implementation via MATLAB Engine
3. k-wave-python: Precompiled C++ binaries with Python API

Mathematical Foundation:
- All three implementations solve the same first-order acoustic equations
- k-Wave uses k-space PSTD (exact spatial derivatives in Fourier space)
- pykwavers uses FDTD (2nd-8th order) or PSTD (spectral accuracy)
- Comparison metrics validate numerical accuracy and performance

Validation Criteria (Sprint 217 specifications):
- L2 error < 0.01 (1% relative error)
- Linf error < 0.05 (5% relative error)
- Phase error < 0.1 rad
- Arrival time error < 1% (for plane waves)

Architecture:
- Clean separation: Configuration → Execution → Analysis → Visualization
- Unidirectional dependencies: no circular imports
- Domain-driven: SimulationConfig, ComparisonResult, ValidationReport
- Pure functions: stateless comparison and metric computation

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-04
Sprint: 217 Session 10 - Unified Comparison Framework
"""

import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# pykwavers imports
try:
    import pykwavers as kw

    PYKWAVERS_AVAILABLE = True
except ImportError:
    PYKWAVERS_AVAILABLE = False
    warnings.warn("pykwavers not available. Install from source.", UserWarning)

# k-wave-python bridge
try:
    from .kwave_python_bridge import (
        KWAVE_PYTHON_AVAILABLE,
        GridParams,
        KWavePythonBridge,
        MediumParams,
        SensorParams,
        SourceParams,
        compute_error_metrics,
    )
except ImportError:
    KWAVE_PYTHON_AVAILABLE = False
    warnings.warn("k-wave-python bridge not available.", UserWarning)

# k-Wave MATLAB bridge
try:
    from .kwave_bridge import MATLAB_AVAILABLE, KWaveBridge

    KWAVE_MATLAB_AVAILABLE = MATLAB_AVAILABLE
except ImportError:
    KWAVE_MATLAB_AVAILABLE = False
    warnings.warn("k-Wave MATLAB bridge not available.", UserWarning)


# ============================================================================
# Domain Models: Comparison Configuration
# ============================================================================


class SimulatorType(Enum):
    """Available simulator implementations."""

    PYKWAVERS_FDTD = "pykwavers_fdtd"
    PYKWAVERS_PSTD = "pykwavers_pstd"
    PYKWAVERS_HYBRID = "pykwavers_hybrid"
    KWAVE_MATLAB = "kwave_matlab"
    KWAVE_PYTHON = "kwave_python"


@dataclass
class SimulationConfig:
    """
    Unified simulation configuration for all simulators.

    This configuration is simulator-agnostic and gets translated to
    simulator-specific formats by adapter functions.

    Attributes:
        grid_shape: (Nx, Ny, Nz)
        grid_spacing: (dx, dy, dz) [m]
        sound_speed: Scalar or 3D array [m/s]
        density: Scalar or 3D array [kg/m^3]
        absorption_coeff: alpha_0 [dB/(MHz^y*cm)]
        absorption_power: y (power law exponent)
        source_frequency: [Hz]
        source_amplitude: [Pa]
        source_position: (x, y, z) or None for plane wave
        sensor_position: (x, y, z) for point sensor
        duration: [s]
        dt: Time step [s] (None for auto)
        pml_size: PML layer thickness [grid points]
    """

    grid_shape: Tuple[int, int, int]
    grid_spacing: Tuple[float, float, float]  # [m]
    sound_speed: Union[float, NDArray[np.float64]]  # [m/s]
    density: Union[float, NDArray[np.float64]]  # [kg/m³]
    source_frequency: float  # [Hz]
    source_amplitude: float  # [Pa]
    duration: float  # [s]
    source_position: Optional[Tuple[float, float, float]] = None  # None = plane wave
    sensor_position: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    absorption_coeff: float = 0.0  # [dB/(MHz^y·cm)]
    absorption_power: float = 1.5
    dt: Optional[float] = None  # [s], None for auto
    pml_size: int = 20
    cfl: float = 0.3

    @property
    def num_time_steps(self) -> int:
        """Compute number of time steps."""
        if self.dt is None:
            c_max = (
                float(np.max(self.sound_speed))
                if isinstance(self.sound_speed, np.ndarray)
                else self.sound_speed
            )
            dx_min = min(self.grid_spacing)
            dt = self.cfl * dx_min / c_max
        else:
            dt = self.dt
        return int(self.duration / dt)

    @property
    def wavelength(self) -> float:
        """Compute wavelength [m]."""
        c = (
            float(np.mean(self.sound_speed))
            if isinstance(self.sound_speed, np.ndarray)
            else self.sound_speed
        )
        return c / self.source_frequency

    @property
    def points_per_wavelength(self) -> float:
        """Compute points per wavelength."""
        dx = self.grid_spacing[0]
        return self.wavelength / dx


@dataclass
class SimulationResult:
    """
    Results from a single simulator.

    Attributes:
        simulator: Simulator type
        pressure: Pressure time series [Pa] (shape: [nt])
        time: Time array [s]
        execution_time: Runtime [s]
        memory_usage: Peak memory [bytes] (optional)
        metadata: Additional simulator-specific data
    """

    simulator: SimulatorType
    pressure: NDArray[np.float64]
    time: NDArray[np.float64]
    execution_time: float
    memory_usage: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """
    Comparison results between simulators.

    Attributes:
        config: Simulation configuration
        results: Dictionary of simulator results
        reference: Reference simulator (for error computation)
        error_metrics: Dictionary of error metrics vs reference
        validation_passed: Whether validation criteria passed
        validation_report: Detailed validation report
    """

    config: SimulationConfig
    results: Dict[SimulatorType, SimulationResult]
    reference: SimulatorType
    error_metrics: Dict[SimulatorType, Dict[str, float]]
    validation_passed: Dict[SimulatorType, bool]
    validation_report: str


# ============================================================================
# Adapter Functions: Configuration Translation
# ============================================================================


def config_to_pykwavers(config: SimulationConfig) -> Tuple:
    """
    Convert unified config to pykwavers objects.

    Returns:
        (grid, medium, source, sensor, nt, dt)
    """
    if not PYKWAVERS_AVAILABLE:
        raise RuntimeError("pykwavers not available")

    # Grid
    grid = kw.Grid(
        nx=config.grid_shape[0],
        ny=config.grid_shape[1],
        nz=config.grid_shape[2],
        dx=config.grid_spacing[0],
        dy=config.grid_spacing[1],
        dz=config.grid_spacing[2],
    )

    # Medium
    medium = kw.Medium.homogeneous(
        sound_speed=float(config.sound_speed)
        if isinstance(config.sound_speed, (int, float))
        else config.sound_speed,
        density=float(config.density)
        if isinstance(config.density, (int, float))
        else config.density,
        absorption=config.absorption_coeff,
    )

    # Source
    if config.source_position is None:
        # Plane wave - use custom mask to match k-wave-python boundary condition
        # k-wave-python applies uniform signal at z=0, not a spatially varying plane wave
        nt = config.num_time_steps
        c_max = (
            float(np.max(config.sound_speed))
            if isinstance(config.sound_speed, np.ndarray)
            else config.sound_speed
        )
        dx_min = min(config.grid_spacing)
        dt_actual = config.dt if config.dt is not None else (config.cfl * dx_min / c_max)

        # Create mask at z=0 (matches k-wave-python setup)
        mask = np.zeros(config.grid_shape, dtype=np.float64)
        mask[:, :, 0] = 1.0

        # Create time signal
        t = np.arange(nt) * dt_actual
        signal = config.source_amplitude * np.sin(2 * np.pi * config.source_frequency * t)

        source = kw.Source.from_mask(mask, signal, frequency=config.source_frequency)
    else:
        # Point source - use mask + signal to match k-wave-python mass source handling
        nt = config.num_time_steps
        c_max = (
            float(np.max(config.sound_speed))
            if isinstance(config.sound_speed, np.ndarray)
            else config.sound_speed
        )
        dx_min = min(config.grid_spacing)
        dt_actual = config.dt if config.dt is not None else (config.cfl * dx_min / c_max)

        mask = np.zeros(config.grid_shape, dtype=np.float64)
        ix = int(config.source_position[0] / config.grid_spacing[0])
        iy = int(config.source_position[1] / config.grid_spacing[1])
        iz = int(config.source_position[2] / config.grid_spacing[2])
        ix = max(0, min(config.grid_shape[0] - 1, ix))
        iy = max(0, min(config.grid_shape[1] - 1, iy))
        iz = max(0, min(config.grid_shape[2] - 1, iz))
        mask[ix, iy, iz] = 1.0

        t = np.arange(nt) * dt_actual
        signal = config.source_amplitude * np.sin(2 * np.pi * config.source_frequency * t)

        source = kw.Source.from_mask(mask, signal, frequency=config.source_frequency)

    # Sensor
    sensor = kw.Sensor.point(position=config.sensor_position)

    # Time parameters
    nt = config.num_time_steps
    dt = config.dt

    return grid, medium, source, sensor, nt, dt


def config_to_kwave_python(config: SimulationConfig) -> Tuple:
    """
    Convert unified config to k-wave-python parameters.

    Returns:
        (GridParams, MediumParams, SourceParams, SensorParams, nt)
    """
    if not KWAVE_PYTHON_AVAILABLE:
        raise RuntimeError("k-wave-python not available")

    # Grid
    grid = GridParams(
        Nx=config.grid_shape[0],
        Ny=config.grid_shape[1],
        Nz=config.grid_shape[2],
        dx=config.grid_spacing[0],
        dy=config.grid_spacing[1],
        dz=config.grid_spacing[2],
        dt=config.dt,
        pml_size=config.pml_size,
    )

    # Medium
    medium = MediumParams(
        sound_speed=config.sound_speed,
        density=config.density,
        alpha_coeff=config.absorption_coeff,
        alpha_power=config.absorption_power,
    )

    # Source
    nt = config.num_time_steps
    c_max = (
        float(np.max(config.sound_speed))
        if isinstance(config.sound_speed, np.ndarray)
        else config.sound_speed
    )
    dt = grid.compute_stable_dt(c_max, cfl=config.cfl) if config.dt is None else config.dt

    if config.source_position is None:
        # Plane wave source
        p_mask = np.zeros(config.grid_shape, dtype=bool)
        p_mask[:, :, 0] = True  # Source at z=0
        t = np.arange(nt) * dt
        p_signal = config.source_amplitude * np.sin(2 * np.pi * config.source_frequency * t)
    else:
        # Point source
        p_mask = np.zeros(config.grid_shape, dtype=bool)
        ix = int(config.source_position[0] / config.grid_spacing[0])
        iy = int(config.source_position[1] / config.grid_spacing[1])
        iz = int(config.source_position[2] / config.grid_spacing[2])
        p_mask[ix, iy, iz] = True
        t = np.arange(nt) * dt
        p_signal = config.source_amplitude * np.sin(2 * np.pi * config.source_frequency * t)

    source = SourceParams(
        p_mask=p_mask,
        p=p_signal,
        frequency=config.source_frequency,
        amplitude=config.source_amplitude,
    )

    # Sensor
    sensor_mask = np.zeros(config.grid_shape, dtype=bool)
    ix = int(config.sensor_position[0] / config.grid_spacing[0])
    iy = int(config.sensor_position[1] / config.grid_spacing[1])
    iz = int(config.sensor_position[2] / config.grid_spacing[2])
    sensor_mask[ix, iy, iz] = True

    sensor = SensorParams(mask=sensor_mask, record=["p"])

    return grid, medium, source, sensor, nt


# ============================================================================
# Execution Functions: Run Simulators
# ============================================================================


def run_pykwavers(config: SimulationConfig, solver_type: str = "fdtd") -> SimulationResult:
    """
    Run pykwavers simulation.

    Args:
        config: Simulation configuration
        solver_type: 'fdtd', 'pstd', or 'hybrid'

    Returns:
        SimulationResult
    """
    if not PYKWAVERS_AVAILABLE:
        raise RuntimeError("pykwavers not available")

    grid, medium, source, sensor, nt, dt = config_to_pykwavers(config)

    # Map solver type string to SolverType enum
    solver_type_map = {
        "fdtd": kw.SolverType.FDTD,
        "pstd": kw.SolverType.PSTD,
        "hybrid": kw.SolverType.Hybrid,
    }

    if solver_type not in solver_type_map:
        raise ValueError(f"Unknown solver_type: {solver_type}. Must be 'fdtd', 'pstd', or 'hybrid'")

    start_time = time.perf_counter()

    # Pass solver type to Simulation constructor (not to run())
    sim = kw.Simulation(
        grid,
        medium,
        source,
        sensor,
        solver=solver_type_map[solver_type],
        pml_size=config.pml_size,
    )
    result = sim.run(time_steps=nt, dt=dt)

    execution_time = time.perf_counter() - start_time

    # Map solver type to simulator enum for results
    simulator_map = {
        "fdtd": SimulatorType.PYKWAVERS_FDTD,
        "pstd": SimulatorType.PYKWAVERS_PSTD,
        "hybrid": SimulatorType.PYKWAVERS_HYBRID,
    }

    return SimulationResult(
        simulator=simulator_map[solver_type],
        pressure=result.sensor_data.flatten(),
        time=result.time,
        execution_time=execution_time,
        metadata={"solver": solver_type, "dt": result.dt, "nt": result.time_steps},
    )


def run_kwave_python(config: SimulationConfig) -> SimulationResult:
    """
    Run k-wave-python simulation.

    Args:
        config: Simulation configuration

    Returns:
        SimulationResult
    """
    if not KWAVE_PYTHON_AVAILABLE:
        raise RuntimeError("k-wave-python not available")

    grid, medium, source, sensor, nt = config_to_kwave_python(config)

    bridge = KWavePythonBridge(cache_dir="./kwave_cache")
    result = bridge.run_simulation(grid, medium, source, sensor, nt)

    return SimulationResult(
        simulator=SimulatorType.KWAVE_PYTHON,
        pressure=result.sensor_data.flatten(),
        time=result.time_array,
        execution_time=result.execution_time,
        metadata={"nt": nt, "dt": grid.dt},
    )


def run_kwave_matlab(config: SimulationConfig) -> SimulationResult:
    """
    Run k-Wave MATLAB simulation via the KWaveBridge.

    Args:
        config: Simulation configuration

    Returns:
        SimulationResult
    """
    if not KWAVE_MATLAB_AVAILABLE:
        raise RuntimeError("k-Wave MATLAB not available")

    from .kwave_bridge import GridConfig, MediumConfig, SourceConfig, SensorConfig

    # Convert SimulationConfig to k-Wave bridge types
    grid = GridConfig(
        Nx=config.grid_shape[0],
        Ny=config.grid_shape[1],
        Nz=config.grid_shape[2],
        dx=config.grid_spacing[0],
        dy=config.grid_spacing[1],
        dz=config.grid_spacing[2],
        pml_size=config.pml_size,
    )

    medium = MediumConfig(
        sound_speed=float(config.sound_speed)
        if not isinstance(config.sound_speed, np.ndarray)
        else config.sound_speed,
        density=float(config.density)
        if not isinstance(config.density, np.ndarray)
        else config.density,
        alpha_coeff=config.absorption_coeff,
        alpha_power=config.absorption_power,
    )

    # Build source mask and signal
    source_mask = np.zeros(config.grid_shape, dtype=np.float64)
    if config.source_position is None:
        # Plane wave source at z=0
        source_mask[:, :, 0] = 1.0
    else:
        ix = int(config.source_position[0] / config.grid_spacing[0])
        iy = int(config.source_position[1] / config.grid_spacing[1])
        iz = int(config.source_position[2] / config.grid_spacing[2])
        ix = max(0, min(config.grid_shape[0] - 1, ix))
        iy = max(0, min(config.grid_shape[1] - 1, iy))
        iz = max(0, min(config.grid_shape[2] - 1, iz))
        source_mask[ix, iy, iz] = 1.0

    c_max = (
        float(np.max(config.sound_speed))
        if isinstance(config.sound_speed, np.ndarray)
        else config.sound_speed
    )
    dx_min = min(config.grid_spacing)
    dt_actual = config.dt if config.dt is not None else (config.cfl * dx_min / c_max)
    nt = config.num_time_steps

    t = np.arange(nt) * dt_actual
    p_signal = config.source_amplitude * np.sin(2 * np.pi * config.source_frequency * t)

    source = SourceConfig(p_mask=source_mask, p_signal=p_signal)

    # Build sensor mask
    sensor_mask = np.zeros(config.grid_shape, dtype=np.float64)
    sx = int(config.sensor_position[0] / config.grid_spacing[0])
    sy = int(config.sensor_position[1] / config.grid_spacing[1])
    sz = int(config.sensor_position[2] / config.grid_spacing[2])
    sx = max(0, min(config.grid_shape[0] - 1, sx))
    sy = max(0, min(config.grid_shape[1] - 1, sy))
    sz = max(0, min(config.grid_shape[2] - 1, sz))
    sensor_mask[sx, sy, sz] = 1.0

    sensor = SensorConfig(mask=sensor_mask)

    bridge = KWaveBridge()
    result = bridge.run_simulation(grid, medium, source, sensor, nt=nt, dt=dt_actual)

    return SimulationResult(
        simulator=SimulatorType.KWAVE_MATLAB,
        pressure=result.sensor_data.flatten() if result.sensor_data is not None else np.array([]),
        time=result.time_array,
        execution_time=result.execution_time,
        metadata={"nt": nt, "dt": dt_actual},
    )


# ============================================================================
# Comparison Functions: Multi-Simulator Comparison
# ============================================================================


def run_comparison(
    config: SimulationConfig,
    simulators: List[SimulatorType],
    reference: Optional[SimulatorType] = None,
) -> ComparisonResult:
    """
    Run comparison across multiple simulators.

    Args:
        config: Simulation configuration
        simulators: List of simulators to run
        reference: Reference simulator for error computation (default: k-wave-python)

    Returns:
        ComparisonResult with all results and metrics
    """
    print("=" * 80)
    print("Multi-Simulator Comparison")
    print("=" * 80)
    print(f"Grid: {config.grid_shape}")
    print(f"Spacing: {tuple(s * 1e3 for s in config.grid_spacing)} mm")
    print(f"Duration: {config.duration * 1e6:.1f} us ({config.num_time_steps} steps)")
    print(
        f"Source: {config.source_frequency * 1e-6:.1f} MHz, {config.source_amplitude * 1e-3:.0f} kPa"
    )
    print(f"Wavelength: {config.wavelength * 1e3:.2f} mm ({config.points_per_wavelength:.1f} PPW)")
    print()

    # Run all simulators
    results = {}
    for sim_type in simulators:
        print(f"Running {sim_type.value}...")
        try:
            if sim_type in [
                SimulatorType.PYKWAVERS_FDTD,
                SimulatorType.PYKWAVERS_PSTD,
                SimulatorType.PYKWAVERS_HYBRID,
            ]:
                solver_map = {
                    SimulatorType.PYKWAVERS_FDTD: "fdtd",
                    SimulatorType.PYKWAVERS_PSTD: "pstd",
                    SimulatorType.PYKWAVERS_HYBRID: "hybrid",
                }
                result = run_pykwavers(config, solver_type=solver_map[sim_type])
            elif sim_type == SimulatorType.KWAVE_PYTHON:
                result = run_kwave_python(config)
            elif sim_type == SimulatorType.KWAVE_MATLAB:
                result = run_kwave_matlab(config)
            else:
                raise ValueError(f"Unknown simulator type: {sim_type}")

            results[sim_type] = result
            print(f"  [OK] Completed in {result.execution_time:.3f}s")
        except Exception as e:
            print(f"  [X] Failed: {e}")
            continue

    if not results:
        raise RuntimeError("No simulators succeeded")

    # Set reference (default: k-wave-python if available)
    if reference is None:
        if SimulatorType.KWAVE_PYTHON in results:
            reference = SimulatorType.KWAVE_PYTHON
        else:
            reference = list(results.keys())[0]

    print()
    print(f"Reference simulator: {reference.value}")
    print()

    # Compute error metrics
    ref_result = results[reference]
    error_metrics = {}
    validation_passed = {}

    for sim_type, result in results.items():
        if sim_type == reference:
            continue

        metrics = compute_error_metrics(ref_result.pressure, result.pressure)
        error_metrics[sim_type] = metrics

        # Validation against acceptance criteria
        l2_pass = metrics["l2_error"] < 0.01
        linf_pass = metrics["linf_error"] < 0.05
        validation_passed[sim_type] = l2_pass and linf_pass

    # Generate validation report
    report = _generate_validation_report(results, reference, error_metrics, validation_passed)

    return ComparisonResult(
        config=config,
        results=results,
        reference=reference,
        error_metrics=error_metrics,
        validation_passed=validation_passed,
        validation_report=report,
    )


def _generate_validation_report(
    results: Dict[SimulatorType, SimulationResult],
    reference: SimulatorType,
    error_metrics: Dict[SimulatorType, Dict[str, float]],
    validation_passed: Dict[SimulatorType, bool],
) -> str:
    """Generate comprehensive validation report."""
    lines = [
        "=" * 80,
        "VALIDATION REPORT",
        "=" * 80,
        "",
        f"Reference: {reference.value}",
        "",
        "Performance Summary:",
        "-" * 80,
    ]

    # Performance table
    for sim_type, result in results.items():
        speedup = (
            results[reference].execution_time / result.execution_time
            if sim_type != reference
            else 1.0
        )
        lines.append(
            f"{sim_type.value:25s} {result.execution_time:8.3f}s  ({speedup:5.2f}x vs reference)"
        )

    lines.extend(["", "Accuracy Metrics:", "-" * 80])

    # Accuracy table
    for sim_type, metrics in error_metrics.items():
        passed = "[OK] PASS" if validation_passed[sim_type] else "[X] FAIL"
        lines.extend(
            [
                f"{sim_type.value}:",
                f"  L2 error:     {metrics['l2_error']:.2e}  "
                f"{'[OK]' if metrics['l2_error'] < 0.01 else '[X]'} (< 0.01)",
                f"  Linf error:   {metrics['linf_error']:.2e}  "
                f"{'[OK]' if metrics['linf_error'] < 0.05 else '[X]'} (< 0.05)",
                f"  RMSE:         {metrics['rmse']:.2e}",
                f"  Max error:    {metrics['max_abs_error']:.2e}",
                f"  Correlation:  {metrics['correlation']:.4f}",
                f"  Overall:      {passed}",
                "",
            ]
        )

    lines.extend(["=" * 80])

    return "\n".join(lines)


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_comparison(comparison: ComparisonResult, output_path: Optional[Path] = None) -> None:
    """
    Plot comparison results.

    Args:
        comparison: ComparisonResult object
        output_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Time series overlay
    ax1 = axes[0]
    for sim_type, result in comparison.results.items():
        label = sim_type.value
        linestyle = "-" if sim_type == comparison.reference else "--"
        linewidth = 2 if sim_type == comparison.reference else 1.5
        alpha = 1.0 if sim_type == comparison.reference else 0.7
        ax1.plot(
            result.time * 1e6,
            result.pressure / 1e3,
            label=label,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax1.set_xlabel("Time [us]")
    ax1.set_ylabel("Pressure [kPa]")
    ax1.set_title("Pressure Time Series Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error vs reference
    ax2 = axes[1]
    ref_result = comparison.results[comparison.reference]
    for sim_type, result in comparison.results.items():
        if sim_type == comparison.reference:
            continue
        min_len = min(len(ref_result.pressure), len(result.pressure))
        error = (result.pressure[:min_len] - ref_result.pressure[:min_len]) / 1e3
        ax2.plot(result.time[:min_len] * 1e6, error, label=sim_type.value, linewidth=1.5)

    ax2.set_xlabel("Time [us]")
    ax2.set_ylabel("Pressure Error [kPa]")
    ax2.set_title(f"Error vs {comparison.reference.value}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)

    # Plot 3: Performance and accuracy bar chart
    ax3 = axes[2]
    simulators = list(comparison.results.keys())
    sim_names = [s.value for s in simulators]

    # Performance (execution time)
    times = [comparison.results[s].execution_time for s in simulators]

    # Accuracy (L2 error)
    errors = []
    for s in simulators:
        if s == comparison.reference:
            errors.append(0)
        else:
            errors.append(comparison.error_metrics[s]["l2_error"])

    x = np.arange(len(sim_names))
    width = 0.35

    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(x - width / 2, times, width, label="Execution Time", color="skyblue")
    bars2 = ax3_twin.bar(x + width / 2, errors, width, label="L2 Error", color="salmon")

    ax3.set_xlabel("Simulator")
    ax3.set_ylabel("Execution Time [s]", color="skyblue")
    ax3_twin.set_ylabel("L2 Error", color="salmon")
    ax3.set_title("Performance and Accuracy Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(sim_names, rotation=15, ha="right")
    ax3.tick_params(axis="y", labelcolor="skyblue")
    ax3_twin.tick_params(axis="y", labelcolor="salmon")
    ax3.grid(True, alpha=0.3, axis="y")

    # Add legends
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {output_path}")

    plt.show()


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Example usage: plane wave comparison."""
    print("=" * 80)
    print("pykwavers vs k-Wave Comparison Framework")
    print("=" * 80)
    print()

    # Check availability
    print("Available simulators:")
    print(f"  pykwavers:     {'[OK]' if PYKWAVERS_AVAILABLE else '[X]'}")
    print(f"  k-wave-python: {'[OK]' if KWAVE_PYTHON_AVAILABLE else '[X]'}")
    print(f"  k-Wave MATLAB: {'[OK]' if KWAVE_MATLAB_AVAILABLE else '[X]'}")
    print()

    if not PYKWAVERS_AVAILABLE:
        print("[X] pykwavers not available. Cannot run comparison.")
        return

    # Create test configuration
    config = SimulationConfig(
        grid_shape=(64, 64, 64),
        grid_spacing=(0.1e-3, 0.1e-3, 0.1e-3),  # 0.1 mm
        sound_speed=1500.0,  # m/s
        density=1000.0,  # kg/m^3
        source_frequency=1e6,  # 1 MHz
        source_amplitude=1e5,  # 100 kPa
        duration=10e-6,  # 10 us
        source_position=None,  # Plane wave
        sensor_position=(3.2e-3, 3.2e-3, 3.2e-3),  # Center
        pml_size=10,
    )

    # Select simulators
    simulators = [SimulatorType.PYKWAVERS_FDTD]
    if KWAVE_PYTHON_AVAILABLE:
        simulators.append(SimulatorType.KWAVE_PYTHON)

    # Run comparison
    try:
        comparison = run_comparison(config, simulators)
        print(comparison.validation_report)

        # Plot results
        plot_comparison(comparison, output_path=Path("./comparison_results.png"))

    except Exception as e:
        print(f"[X] Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()
