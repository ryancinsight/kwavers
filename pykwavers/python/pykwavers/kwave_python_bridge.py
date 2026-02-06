#!/usr/bin/env python3
"""
k-wave-python Bridge for Automated Comparison and Validation

This module provides a Python interface to k-wave-python (waltsims/k-wave-python),
which uses precompiled C++ k-Wave binaries for fast, validated acoustic simulations.

Mathematical Specifications:
- k-Wave uses k-space pseudospectral time domain (k-space PSTD) method
- Spatial derivatives computed exactly in Fourier space (zero numerical dispersion)
- Perfectly matched layers (PML) for boundary absorption
- Power-law absorption: α(ω) = α₀|ω|^y (Szabo 1994)
- Nonlinear propagation via Westervelt equation (optional)

Key Differences from MATLAB k-Wave Bridge:
- Uses precompiled C++ binaries (no MATLAB required)
- Direct Python API (no engine overhead)
- HDF5-based data I/O
- Identical numerical results to MATLAB k-Wave

References:
1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
   and reconstruction of photoacoustic wave fields." J. Biomed. Opt., 15(2), 021314.
2. Jaros, J., et al. (2016). "Full-wave nonlinear ultrasound simulation on distributed
   clusters with applications in high-intensity focused ultrasound." Int. J. HPC Apps.
3. waltsims/k-wave-python: https://github.com/waltsims/k-wave-python

Architecture:
- Clean separation between configuration (dataclasses) and execution (bridge)
- Unidirectional dependencies: pykwavers → k-wave-python (no circular deps)
- Domain-driven design: GridParams, MediumParams, SourceParams, SensorParams
- Validation: All inputs verified against mathematical constraints

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-04
Sprint: 217 Session 10 - k-wave-python Integration
"""

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# k-wave-python imports (graceful degradation if not installed)
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    KWAVE_PYTHON_AVAILABLE = True
except ImportError:
    KWAVE_PYTHON_AVAILABLE = False
    warnings.warn(
        "k-wave-python not available. Install with: pip install k-wave-python\n"
        "Comparison and validation features will be disabled.",
        UserWarning,
    )


# ============================================================================
# Domain Models: Configuration Parameters
# ============================================================================


@dataclass(frozen=True)
class GridParams:
    """
    Immutable grid configuration for k-Wave simulations.

    Mathematical Specification:
    - Uniform Cartesian grid with spacing dx, dy, dz
    - Domain size: Lx = Nx·dx, Ly = Ny·dy, Lz = Nz·dz
    - Grid points include boundaries
    - CFL stability: dt ≤ CFL·dx/c_max, CFL ≈ 0.3 (conservative)

    Invariants:
    - Nx, Ny, Nz > 0
    - dx, dy, dz > 0
    - pml_size ≥ 0 (typically 10-20 for absorbing boundaries)
    - dt > 0 or None (auto-computed from CFL if None)

    Attributes:
        Nx, Ny, Nz: Number of grid points in each direction
        dx, dy, dz: Grid spacing [m]
        dt: Time step [s] (None for auto-computation)
        pml_size: PML layer thickness [grid points]
        pml_alpha: PML frequency scaling factor (default 2.0)
        pml_inside: Place PML inside (True) or outside (False) domain
    """

    Nx: int
    Ny: int
    Nz: int
    dx: float  # [m]
    dy: float  # [m]
    dz: float  # [m]
    dt: Optional[float] = None  # [s], None for auto
    pml_size: int = 20
    pml_alpha: float = 2.0
    pml_inside: bool = True

    def __post_init__(self):
        """Validate grid parameters against mathematical constraints."""
        if self.Nx <= 0 or self.Ny <= 0 or self.Nz <= 0:
            raise ValueError(
                f"Grid dimensions must be positive: got ({self.Nx}, {self.Ny}, {self.Nz})"
            )
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError(
                f"Grid spacing must be positive: got ({self.dx}, {self.dy}, {self.dz})"
            )
        if self.dt is not None and self.dt <= 0:
            raise ValueError(f"Time step must be positive: got {self.dt}")
        if self.pml_size < 0:
            raise ValueError(f"PML size must be non-negative: got {self.pml_size}")
        if self.pml_alpha <= 0:
            raise ValueError(f"PML alpha must be positive: got {self.pml_alpha}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (Nx, Ny, Nz)."""
        return (self.Nx, self.Ny, self.Nz)

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Grid spacing (dx, dy, dz) [m]."""
        return (self.dx, self.dy, self.dz)

    @property
    def domain_size(self) -> Tuple[float, float, float]:
        """Physical domain size (Lx, Ly, Lz) [m]."""
        return (self.Nx * self.dx, self.Ny * self.dy, self.Nz * self.dz)

    @property
    def total_points(self) -> int:
        """Total number of grid points."""
        return self.Nx * self.Ny * self.Nz

    def compute_stable_dt(self, c_max: float, cfl: float = 0.3) -> float:
        """
        Compute stable time step from CFL condition.

        Mathematical Specification:
        dt ≤ CFL · min(dx, dy, dz) / c_max

        Args:
            c_max: Maximum sound speed in domain [m/s]
            cfl: CFL number (default 0.3, conservative)

        Returns:
            Stable time step [s]
        """
        dx_min = min(self.dx, self.dy, self.dz)
        return cfl * dx_min / c_max


@dataclass
class MediumParams:
    """
    Acoustic medium properties for k-Wave simulations.

    Mathematical Specification (Linear Acoustics):
    - Sound speed: c₀(x) [m/s]
    - Density: ρ₀(x) [kg/m³]
    - Absorption: α(ω) = α₀|ω|^y [Np/m] (power law, Szabo 1994)
      - α₀: absorption coefficient [dB/(MHz^y·cm)]
      - y: power law exponent (typically 1.0-2.0)
    - Nonlinearity: B/A parameter (optional)

    Invariants:
    - c₀ > 0 (everywhere if heterogeneous)
    - ρ₀ > 0 (everywhere if heterogeneous)
    - α₀ ≥ 0
    - 0 ≤ y ≤ 3 (physical range)

    Attributes:
        sound_speed: Scalar or 3D array [m/s]
        density: Scalar or 3D array [kg/m³]
        alpha_coeff: Absorption coefficient α₀ [dB/(MHz^y·cm)]
        alpha_power: Power law exponent y
        BonA: B/A nonlinearity parameter (0 for linear)
    """

    sound_speed: Union[float, NDArray[np.float64]]  # [m/s]
    density: Union[float, NDArray[np.float64]]  # [kg/m³]
    alpha_coeff: float = 0.0  # [dB/(MHz^y·cm)]
    alpha_power: float = 1.5  # dimensionless
    BonA: float = 0.0  # B/A nonlinearity parameter

    def __post_init__(self):
        """Validate medium parameters against physical constraints."""
        # Validate sound speed
        if isinstance(self.sound_speed, (int, float)):
            if self.sound_speed <= 0:
                raise ValueError(f"Sound speed must be positive: got {self.sound_speed}")
        elif isinstance(self.sound_speed, np.ndarray):
            if np.any(self.sound_speed <= 0):
                raise ValueError("Sound speed must be positive everywhere")
        else:
            raise TypeError(f"Sound speed must be float or ndarray: got {type(self.sound_speed)}")

        # Validate density
        if isinstance(self.density, (int, float)):
            if self.density <= 0:
                raise ValueError(f"Density must be positive: got {self.density}")
        elif isinstance(self.density, np.ndarray):
            if np.any(self.density <= 0):
                raise ValueError("Density must be positive everywhere")
        else:
            raise TypeError(f"Density must be float or ndarray: got {type(self.density)}")

        # Validate absorption
        if self.alpha_coeff < 0:
            raise ValueError(f"Absorption coefficient must be non-negative: got {self.alpha_coeff}")
        if self.alpha_power < 0 or self.alpha_power > 3:
            raise ValueError(f"Alpha power must be in [0, 3]: got {self.alpha_power}")

    @property
    def is_homogeneous(self) -> bool:
        """Check if medium is homogeneous (scalar properties)."""
        return isinstance(self.sound_speed, (int, float)) and isinstance(self.density, (int, float))

    @property
    def is_absorbing(self) -> bool:
        """Check if medium has absorption."""
        return self.alpha_coeff > 0

    @property
    def is_nonlinear(self) -> bool:
        """Check if medium is nonlinear."""
        return self.BonA != 0

    def validate_shape(self, grid: GridParams) -> None:
        """
        Validate heterogeneous medium arrays match grid shape.

        Args:
            grid: Grid configuration

        Raises:
            ValueError: If array shapes don't match grid
        """
        if isinstance(self.sound_speed, np.ndarray):
            if self.sound_speed.shape != grid.shape:
                raise ValueError(
                    f"Sound speed shape {self.sound_speed.shape} != grid shape {grid.shape}"
                )
        if isinstance(self.density, np.ndarray):
            if self.density.shape != grid.shape:
                raise ValueError(f"Density shape {self.density.shape} != grid shape {grid.shape}")


@dataclass
class SourceParams:
    """
    Acoustic source configuration for k-Wave simulations.

    Mathematical Specification:
    - Pressure source: p_source(x,t) = mask(x) · signal(t)
    - Velocity source: u_source(x,t) = mask(x) · signal(t)
    - Initial pressure: p₀(x)

    Attributes:
        p_mask: Spatial pressure source mask (3D binary/weighted array)
        p: Temporal pressure signal [Pa] (1D or 2D: [num_sources, nt])
        u_mask: Spatial velocity source mask (optional)
        u: Temporal velocity signal [m/s] (optional)
        p0: Initial pressure field [Pa] (3D array, optional)
        frequency: Source frequency [Hz] (for reference/validation)
        amplitude: Source amplitude [Pa] (for reference/validation)
    """

    p_mask: Optional[NDArray[np.bool_]] = None
    p: Optional[NDArray[np.float64]] = None
    u_mask: Optional[NDArray[np.bool_]] = None
    u: Optional[NDArray[np.float64]] = None
    p0: Optional[NDArray[np.float64]] = None
    frequency: Optional[float] = None
    amplitude: Optional[float] = None

    def __post_init__(self):
        """Validate source configuration."""
        # Check pressure source consistency
        if (self.p_mask is not None) != (self.p is not None):
            raise ValueError("p_mask and p must be both specified or both None")

        # Check velocity source consistency
        if (self.u_mask is not None) != (self.u is not None):
            raise ValueError("u_mask and u must be both specified or both None")

        # At least one source must be specified
        if self.p_mask is None and self.u_mask is None and (self.p0 is None or not np.any(self.p0)):
            raise ValueError("At least one source (p, u, or p0) must be specified")

        # Validate signal dimensions
        if self.p is not None and self.p.ndim not in [1, 2]:
            raise ValueError(f"Pressure signal must be 1D or 2D: got shape {self.p.shape}")
        if self.u is not None and self.u.ndim not in [1, 2]:
            raise ValueError(f"Velocity signal must be 1D or 2D: got shape {self.u.shape}")

    @property
    def has_pressure_source(self) -> bool:
        """Check if pressure source is defined."""
        return self.p_mask is not None and self.p is not None

    @property
    def has_velocity_source(self) -> bool:
        """Check if velocity source is defined."""
        return self.u_mask is not None and self.u is not None

    @property
    def has_initial_pressure(self) -> bool:
        """Check if initial pressure is defined."""
        return self.p0 is not None and np.any(self.p0)

    def validate_shape(self, grid: GridParams) -> None:
        """
        Validate source arrays match grid shape.

        Args:
            grid: Grid configuration

        Raises:
            ValueError: If array shapes don't match grid
        """
        if self.p_mask is not None and self.p_mask.shape != grid.shape:
            raise ValueError(f"p_mask shape {self.p_mask.shape} != grid shape {grid.shape}")
        if self.u_mask is not None and self.u_mask.shape != grid.shape:
            raise ValueError(f"u_mask shape {self.u_mask.shape} != grid shape {grid.shape}")
        if self.p0 is not None and self.p0.shape != grid.shape:
            raise ValueError(f"p0 shape {self.p0.shape} != grid shape {grid.shape}")


@dataclass
class SensorParams:
    """
    Sensor configuration for k-Wave simulations.

    Mathematical Specification:
    - Point sensors: Record field values at discrete points
    - Grid sensors: Record entire field at specified times
    - Cartesian sensors: Binary mask on computational grid

    Attributes:
        mask: Binary mask indicating sensor positions (3D array)
        record: List of fields to record (e.g., ['p', 'u', 'p_max', 'p_rms'])
        record_start_index: Time index to start recording (default 1)
    """

    mask: NDArray[np.bool_]
    record: List[str] = field(default_factory=lambda: ["p"])
    record_start_index: int = 1

    def __post_init__(self):
        """Validate sensor configuration."""
        if not isinstance(self.mask, np.ndarray):
            raise TypeError(f"Sensor mask must be ndarray: got {type(self.mask)}")
        if self.mask.dtype != bool:
            raise TypeError(f"Sensor mask must be boolean: got {self.mask.dtype}")
        if self.mask.ndim != 3:
            raise ValueError(f"Sensor mask must be 3D: got shape {self.mask.shape}")
        if not np.any(self.mask):
            raise ValueError("Sensor mask must have at least one active sensor")
        if self.record_start_index < 1:
            raise ValueError(f"record_start_index must be ≥ 1: got {self.record_start_index}")

    @property
    def num_sensors(self) -> int:
        """Number of active sensor points."""
        return int(np.sum(self.mask))

    def validate_shape(self, grid: GridParams) -> None:
        """
        Validate sensor mask matches grid shape.

        Args:
            grid: Grid configuration

        Raises:
            ValueError: If mask shape doesn't match grid
        """
        if self.mask.shape != grid.shape:
            raise ValueError(f"Sensor mask shape {self.mask.shape} != grid shape {grid.shape}")


@dataclass
class SimulationResult:
    """
    k-Wave simulation results.

    Attributes:
        sensor_data: Recorded sensor data [Pa or m/s] (shape: [num_sensors, nt])
        p_max: Maximum pressure recorded at each sensor [Pa] (optional)
        p_min: Minimum pressure recorded at each sensor [Pa] (optional)
        p_rms: RMS pressure recorded at each sensor [Pa] (optional)
        time_array: Time points [s]
        grid_params: Grid configuration used
        medium_params: Medium configuration used
        execution_time: Simulation runtime [s]
        memory_usage: Peak memory usage [bytes] (optional)
    """

    sensor_data: NDArray[np.float64]
    p_max: Optional[NDArray[np.float64]] = None
    p_min: Optional[NDArray[np.float64]] = None
    p_rms: Optional[NDArray[np.float64]] = None
    time_array: Optional[NDArray[np.float64]] = None
    grid_params: Optional[GridParams] = None
    medium_params: Optional[MediumParams] = None
    execution_time: float = 0.0
    memory_usage: Optional[int] = None

    def __post_init__(self):
        """Validate result data."""
        if self.sensor_data.ndim not in [1, 2]:
            raise ValueError(f"Sensor data must be 1D or 2D: got shape {self.sensor_data.shape}")


# ============================================================================
# Bridge Implementation: k-wave-python Interface
# ============================================================================


class KWavePythonBridge:
    """
    Bridge to k-wave-python for automated acoustic simulation comparison.

    This class provides a clean interface to k-wave-python, enabling:
    1. Direct comparison of pykwavers vs k-Wave results
    2. Validation against k-Wave's extensively tested implementations
    3. Caching of k-Wave results for repeated comparisons
    4. No MATLAB dependency (uses precompiled C++ binaries)

    Mathematical Correctness:
    - k-Wave implements k-space PSTD with exact spatial derivatives
    - Temporal integration: 4th-order Runge-Kutta or predictor-corrector
    - PML boundaries: CPML formulation (Roden & Gedney 2000)
    - Identical numerical results to MATLAB k-Wave

    Architecture:
    - Immutable configuration objects (GridParams, MediumParams, etc.)
    - Pure functions for configuration → k-Wave objects
    - Explicit cache key computation from configuration
    - Unidirectional dependencies: no circular imports

    Usage:
        bridge = KWavePythonBridge(cache_dir="./kwave_cache")
        result = bridge.run_simulation(
            grid=GridParams(Nx=64, Ny=64, Nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3),
            medium=MediumParams(sound_speed=1500.0, density=1000.0),
            source=SourceParams(p_mask=mask, p=signal),
            sensor=SensorParams(mask=sensor_mask),
            nt=1000
        )
    """

    def __init__(self, cache_dir: Optional[Path] = None, enable_cache: bool = True):
        """
        Initialize k-wave-python bridge.

        Args:
            cache_dir: Directory for caching k-Wave results (default: ./kwave_cache)
            enable_cache: Enable result caching (default: True)

        Raises:
            RuntimeError: If k-wave-python is not installed
        """
        if not KWAVE_PYTHON_AVAILABLE:
            raise RuntimeError(
                "k-wave-python not available. Install with:\n"
                "  pip install k-wave-python\n"
                "See: https://github.com/waltsims/k-wave-python"
            )

        self.cache_dir = Path(cache_dir) if cache_dir else Path("./kwave_cache")
        self.enable_cache = enable_cache

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run_simulation(
        self,
        grid: GridParams,
        medium: MediumParams,
        source: SourceParams,
        sensor: SensorParams,
        nt: int,
        simulation_options: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> SimulationResult:
        """
        Run k-Wave acoustic simulation using k-wave-python.

        Mathematical Specification:
        - Solves first-order acoustic equations using k-space PSTD
        - Spatial derivatives: F^{-1}[ik·F[f]] (exact in Fourier space)
        - Temporal integration: 4th-order Runge-Kutta (default)
        - Stability: CFL condition enforced automatically

        Args:
            grid: Grid configuration (immutable)
            medium: Medium properties (immutable)
            source: Source configuration (immutable)
            sensor: Sensor configuration (immutable)
            nt: Number of time steps
            simulation_options: Additional k-Wave options (dict)
            use_cache: Load from cache if available (default: True)

        Returns:
            SimulationResult with pressure data, timing, and metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If simulation fails
        """
        # Validate configurations
        medium.validate_shape(grid)
        source.validate_shape(grid)
        sensor.validate_shape(grid)

        # Compute time step if not specified
        if grid.dt is None:
            if isinstance(medium.sound_speed, np.ndarray):
                c_max = float(np.max(medium.sound_speed))
            else:
                c_max = float(medium.sound_speed)
            dt = grid.compute_stable_dt(c_max, cfl=0.3)
        else:
            dt = grid.dt

        # Check cache
        if self.enable_cache and use_cache:
            cache_key = self._compute_cache_key(grid, medium, source, sensor, nt)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                print(f"Loaded k-Wave result from cache: {cache_key}")
                return cached_result

        # Run simulation
        print(f"Running k-Wave simulation: {grid.shape} grid, {nt} steps...")
        start_time = time.perf_counter()

        try:
            # Create k-Wave objects
            kgrid = self._create_kgrid(grid, medium, nt)
            kmedium = self._create_kmedium(medium, grid)
            ksource = self._create_ksource(source, nt)
            ksensor_obj = self._create_ksensor(sensor)

            # Create simulation options
            sim_options = SimulationOptions(
                pml_inside=grid.pml_inside,
                pml_size=grid.pml_size,
                pml_alpha=grid.pml_alpha,
                data_cast="single",  # Use single precision for speed
                save_to_disk=True,  # Required for CPU simulations
            )

            # Override with user options
            if simulation_options:
                for key, value in simulation_options.items():
                    setattr(sim_options, key, value)

            # Create execution options (required by k-wave-python API)
            exec_options = SimulationExecutionOptions(
                is_gpu_simulation=False,
                verbose_level=0,
                show_sim_log=False,
            )

            # Run k-Wave simulation
            sensor_data = kspaceFirstOrder3D(
                kgrid=kgrid,
                medium=kmedium,
                source=ksource,
                sensor=ksensor_obj,
                simulation_options=sim_options,
                execution_options=exec_options,
            )

            execution_time = time.perf_counter() - start_time

            # Extract results
            result = self._extract_results(
                sensor_data, sensor, grid, medium, nt, dt, execution_time
            )

            # Cache results
            if self.enable_cache:
                self._save_to_cache(cache_key, result)

            print(f"k-Wave simulation complete in {execution_time:.3f}s")
            return result

        except Exception as e:
            raise RuntimeError(f"k-Wave simulation failed: {e}") from e

    def _create_kgrid(self, grid: GridParams, medium: MediumParams, nt: int) -> kWaveGrid:
        """
        Create k-Wave grid object.

        Args:
            grid: Grid parameters
            medium: Medium parameters (for dt computation)
            nt: Number of time steps

        Returns:
            kWaveGrid object
        """
        # Compute time step if not specified
        if grid.dt is None:
            if isinstance(medium.sound_speed, np.ndarray):
                c_max = float(np.max(medium.sound_speed))
            else:
                c_max = float(medium.sound_speed)
            dt = grid.compute_stable_dt(c_max, cfl=0.3)
        else:
            dt = grid.dt

        # Create k-Wave grid
        kgrid = kWaveGrid([grid.Nx, grid.Ny, grid.Nz], [grid.dx, grid.dy, grid.dz])
        kgrid.makeTime(medium.sound_speed, cfl=0.3, t_end=dt * nt)

        return kgrid

    def _create_kmedium(self, medium: MediumParams, grid: GridParams) -> kWaveMedium:
        """
        Create k-Wave medium object.

        Args:
            medium: Medium parameters
            grid: Grid parameters (for shape validation)

        Returns:
            kWaveMedium object
        """
        kmedium = kWaveMedium(sound_speed=medium.sound_speed, density=medium.density)

        # Add absorption if specified
        if medium.is_absorbing:
            kmedium.alpha_coeff = medium.alpha_coeff
            kmedium.alpha_power = medium.alpha_power

        # Add nonlinearity if specified
        if medium.is_nonlinear:
            kmedium.BonA = medium.BonA

        return kmedium

    def _create_ksource(self, source: SourceParams, nt: int) -> kSource:
        """
        Create k-Wave source object.

        Args:
            source: Source parameters
            nt: Number of time steps

        Returns:
            kSource object
        """
        ksource = kSource()

        # Add pressure source
        if source.has_pressure_source:
            ksource.p_mask = source.p_mask
            # Ensure signal is 2D: [num_sources, nt]
            if source.p.ndim == 1:
                # Broadcast to all source points
                num_sources = int(np.sum(source.p_mask))
                ksource.p = np.tile(source.p, (num_sources, 1))
            else:
                ksource.p = source.p

        # Add velocity source
        if source.has_velocity_source:
            ksource.u_mask = source.u_mask
            if source.u.ndim == 1:
                num_sources = int(np.sum(source.u_mask))
                ksource.u = np.tile(source.u, (num_sources, 1))
            else:
                ksource.u = source.u

        # Add initial pressure
        if source.has_initial_pressure:
            ksource.p0 = source.p0

        return ksource

    def _create_ksensor(self, sensor: SensorParams) -> kSensor:
        """
        Create k-Wave sensor object.

        Args:
            sensor: Sensor parameters

        Returns:
            kSensor object
        """
        ksensor_obj = kSensor(sensor.mask)
        ksensor_obj.record = sensor.record
        ksensor_obj.record_start_index = sensor.record_start_index

        return ksensor_obj

    def _extract_results(
        self,
        sensor_data: Dict,
        sensor: SensorParams,
        grid: GridParams,
        medium: MediumParams,
        nt: int,
        dt: float,
        execution_time: float,
    ) -> SimulationResult:
        """
        Extract results from k-Wave output.

        Args:
            sensor_data: k-Wave sensor data dictionary
            sensor: Sensor parameters
            grid: Grid parameters
            medium: Medium parameters
            nt: Number of time steps
            dt: Time step
            execution_time: Simulation runtime

        Returns:
            SimulationResult object
        """
        # Extract pressure data (primary field)
        if isinstance(sensor_data, dict):
            p_data = sensor_data.get("p", sensor_data)
        else:
            p_data = sensor_data

        # Ensure 2D shape [num_sensors, nt]
        if p_data.ndim == 1:
            p_data = p_data.reshape(1, -1)

        # Extract optional fields
        p_max = sensor_data.get("p_max") if isinstance(sensor_data, dict) else None
        p_min = sensor_data.get("p_min") if isinstance(sensor_data, dict) else None
        p_rms = sensor_data.get("p_rms") if isinstance(sensor_data, dict) else None

        # Create time array based on actual data length (k-Wave may return different length)
        actual_nt = p_data.shape[1]
        time_array = np.arange(actual_nt) * dt

        return SimulationResult(
            sensor_data=p_data,
            p_max=p_max,
            p_min=p_min,
            p_rms=p_rms,
            time_array=time_array,
            grid_params=grid,
            medium_params=medium,
            execution_time=execution_time,
        )

    def _compute_cache_key(
        self,
        grid: GridParams,
        medium: MediumParams,
        source: SourceParams,
        sensor: SensorParams,
        nt: int,
    ) -> str:
        """
        Compute unique cache key from simulation configuration.

        Uses SHA256 hash of serialized configuration to ensure:
        - Identical configurations produce same key
        - Different configurations produce different keys
        - Keys are filesystem-safe

        Args:
            grid: Grid parameters
            medium: Medium parameters
            source: Source parameters
            sensor: Sensor parameters
            nt: Number of time steps

        Returns:
            Cache key (hex string)
        """
        config = {
            "grid": {
                "shape": grid.shape,
                "spacing": grid.spacing,
                "dt": grid.dt,
                "pml_size": grid.pml_size,
                "pml_alpha": grid.pml_alpha,
            },
            "medium": {
                "sound_speed": float(medium.sound_speed)
                if isinstance(medium.sound_speed, (int, float))
                else hashlib.sha256(medium.sound_speed.tobytes()).hexdigest(),
                "density": float(medium.density)
                if isinstance(medium.density, (int, float))
                else hashlib.sha256(medium.density.tobytes()).hexdigest(),
                "alpha_coeff": medium.alpha_coeff,
                "alpha_power": medium.alpha_power,
                "BonA": medium.BonA,
            },
            "source": {
                "p_mask_hash": hashlib.sha256(source.p_mask.tobytes()).hexdigest()
                if source.p_mask is not None
                else None,
                "p_hash": hashlib.sha256(source.p.tobytes()).hexdigest()
                if source.p is not None
                else None,
                "frequency": source.frequency,
            },
            "sensor": {
                "mask_hash": hashlib.sha256(sensor.mask.tobytes()).hexdigest(),
                "num_sensors": sensor.num_sensors,
            },
            "nt": nt,
        }

        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _save_to_cache(self, cache_key: str, result: SimulationResult) -> None:
        """
        Save simulation result to cache.

        Args:
            cache_key: Cache key
            result: Simulation result
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"

        np.savez_compressed(
            cache_file,
            sensor_data=result.sensor_data,
            p_max=result.p_max if result.p_max is not None else np.array([]),
            p_min=result.p_min if result.p_min is not None else np.array([]),
            p_rms=result.p_rms if result.p_rms is not None else np.array([]),
            time_array=result.time_array if result.time_array is not None else np.array([]),
            execution_time=result.execution_time,
        )

        print(f"Cached k-Wave result to {cache_file}")

    def _load_from_cache(self, cache_key: str) -> Optional[SimulationResult]:
        """
        Load simulation result from cache.

        Args:
            cache_key: Cache key

        Returns:
            SimulationResult if cached, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if not cache_file.exists():
            return None

        try:
            data = np.load(cache_file)

            result = SimulationResult(
                sensor_data=data["sensor_data"],
                p_max=data["p_max"] if data["p_max"].size > 0 else None,
                p_min=data["p_min"] if data["p_min"].size > 0 else None,
                p_rms=data["p_rms"] if data["p_rms"].size > 0 else None,
                time_array=data["time_array"] if data["time_array"].size > 0 else None,
                execution_time=float(data["execution_time"]),
            )

            return result

        except Exception as e:
            warnings.warn(f"Failed to load cache {cache_file}: {e}", UserWarning)
            return None


# ============================================================================
# Utility Functions: Comparison and Validation
# ============================================================================


def compute_error_metrics(
    reference: NDArray[np.float64], test: NDArray[np.float64]
) -> Dict[str, float]:
    """
    Compute error metrics between reference and test data.

    Mathematical Specifications:
    - L2 error: ||test - ref||_2 / ||ref||_2
    - L∞ error: max|test - ref| / max|ref|
    - RMSE: sqrt(mean((test - ref)²))
    - Max absolute error: max|test - ref|

    Args:
        reference: Reference data (e.g., k-Wave)
        test: Test data (e.g., pykwavers)

    Returns:
        Dictionary of error metrics
    """
    # Ensure same length
    min_len = min(len(reference.flatten()), len(test.flatten()))
    ref = reference.flatten()[:min_len]
    tst = test.flatten()[:min_len]

    # Compute metrics
    l2_error = np.linalg.norm(tst - ref) / np.linalg.norm(ref)
    linf_error = np.max(np.abs(tst - ref)) / np.max(np.abs(ref))
    rmse = np.sqrt(np.mean((tst - ref) ** 2))
    max_abs_error = np.max(np.abs(tst - ref))

    # Correlation coefficient
    correlation = np.corrcoef(ref, tst)[0, 1]

    return {
        "l2_error": l2_error,
        "linf_error": linf_error,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "correlation": correlation,
    }


def validate_against_acceptance_criteria(
    metrics: Dict[str, float], l2_threshold: float = 0.01, linf_threshold: float = 0.05
) -> Tuple[bool, str]:
    """
    Validate error metrics against acceptance criteria.

    Acceptance Criteria (from Sprint 217 specifications):
    - L2 error < 0.01 (1% relative error)
    - L∞ error < 0.05 (5% relative error)

    Args:
        metrics: Error metrics from compute_error_metrics
        l2_threshold: L2 error threshold
        linf_threshold: L∞ error threshold

    Returns:
        (passed, report) tuple
    """
    l2_pass = metrics["l2_error"] < l2_threshold
    linf_pass = metrics["linf_error"] < linf_threshold

    report = f"""
Validation Report:
==================
L2 error:   {metrics["l2_error"]:.2e} {"[OK] PASS" if l2_pass else "[X] FAIL"} (threshold: {l2_threshold:.2e})
L∞ error:   {metrics["linf_error"]:.2e} {"[OK] PASS" if linf_pass else "[X] FAIL"} (threshold: {linf_threshold:.2e})
RMSE:       {metrics["rmse"]:.2e}
Max error:  {metrics["max_abs_error"]:.2e}
Correlation: {metrics["correlation"]:.4f}

Overall: {"[OK] PASS" if (l2_pass and linf_pass) else "[X] FAIL"}
"""

    return l2_pass and linf_pass, report


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("k-wave-python Bridge Test")
    print("=" * 80)

    if not KWAVE_PYTHON_AVAILABLE:
        print("[X] k-wave-python not available. Install with: pip install k-wave-python")
        exit(1)

    # Create test configuration
    grid = GridParams(Nx=64, Ny=64, Nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3, pml_size=10)

    medium = MediumParams(sound_speed=1500.0, density=1000.0, alpha_coeff=0.0)

    # Plane wave source
    p_mask = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool)
    p_mask[:, :, 0] = True  # Source at z=0

    nt = 1000
    dt = grid.compute_stable_dt(1500.0)
    t = np.arange(nt) * dt
    frequency = 1e6  # 1 MHz
    amplitude = 1e5  # 100 kPa
    p_signal = amplitude * np.sin(2 * np.pi * frequency * t)

    source = SourceParams(p_mask=p_mask, p=p_signal, frequency=frequency, amplitude=amplitude)

    # Point sensor at center
    sensor_mask = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool)
    sensor_mask[grid.Nx // 2, grid.Ny // 2, grid.Nz // 2] = True

    sensor = SensorParams(mask=sensor_mask, record=["p"])

    # Run simulation
    try:
        bridge = KWavePythonBridge(cache_dir="./kwave_cache")
        result = bridge.run_simulation(grid, medium, source, sensor, nt)

        print("\nSimulation Results:")
        print(f"  Sensor data shape: {result.sensor_data.shape}")
        print(f"  Execution time: {result.execution_time:.3f}s")
        print(f"  Max pressure: {np.max(np.abs(result.sensor_data)) / 1e3:.2f} kPa")

        print("\n[OK] Test passed!")

    except Exception as e:
        print(f"\n[X] Test failed: {e}")
        raise
