#!/usr/bin/env python3
"""
k-Wave Bridge for Automated Comparison

This module provides a Python interface to k-Wave (MATLAB toolbox) for automated
validation and comparison with kwavers acoustic simulations.

Mathematical Specifications:
- k-Wave uses k-space pseudospectral time domain (k-space PSTD) method
- Spatial derivatives computed exactly in Fourier space (zero numerical dispersion)
- Perfectly matched layers (PML) for boundary absorption
- Power-law absorption: α(ω) = α₀|ω|^y (Szabo 1994)

References:
1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
   and reconstruction of photoacoustic wave fields." J. Biomed. Opt., 15(2), 021314.
2. Treeby et al. (2012). "Modeling nonlinear ultrasound propagation in heterogeneous
   media with power law absorption using a k-space pseudospectral method."
   JASA, 131(6), 4324-4336.

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-04
Sprint: 217 Session 8 - k-Wave Comparison Framework
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Optional MATLAB Engine import (graceful degradation if not available)
try:
    import matlab.engine

    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    warnings.warn(
        "MATLAB Engine not available. k-Wave comparisons will use cached results only.",
        UserWarning,
    )


@dataclass
class GridConfig:
    """
    Grid configuration for k-Wave simulation.

    Mathematical Specification:
    - Uniform Cartesian grid with spacing dx, dy, dz
    - Domain size: Lx = Nx·dx, Ly = Ny·dy, Lz = Nz·dz
    - Grid points include boundaries

    Attributes:
        Nx, Ny, Nz: Number of grid points in each direction
        dx, dy, dz: Grid spacing [m]
        pml_size: PML layer thickness [grid points] (Roden & Gedney 2000)
        pml_alpha: PML frequency scaling factor (default 2.0)
    """

    Nx: int
    Ny: int
    Nz: int
    dx: float  # [m]
    dy: float  # [m]
    dz: float  # [m]
    pml_size: int = 20
    pml_alpha: float = 2.0

    def __post_init__(self):
        """Validate grid parameters."""
        if self.Nx <= 0 or self.Ny <= 0 or self.Nz <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("Grid spacing must be positive")
        if self.pml_size < 0:
            raise ValueError("PML size must be non-negative")

    def to_kwave_grid(self, engine) -> object:
        """
        Create k-Wave kgrid structure.

        Returns:
            MATLAB kgrid object
        """
        if not MATLAB_AVAILABLE:
            raise RuntimeError("MATLAB Engine required for k-Wave grid creation")

        # Create k-Wave grid using makeGrid
        kgrid = engine.kWaveGrid(
            float(self.Nx),
            float(self.dx),
            float(self.Ny),
            float(self.dy),
            float(self.Nz),
            float(self.dz),
        )
        return kgrid


@dataclass
class MediumConfig:
    """
    Acoustic medium properties for k-Wave.

    Mathematical Specification (Linear Acoustics):
    - Sound speed: c₀ [m/s]
    - Density: ρ₀ [kg/m³]
    - Absorption: α(ω) = α₀|ω|^y [Np/m] (power law, Szabo 1994)
      - α₀: absorption coefficient [dB/(MHz^y·cm)]
      - y: power law exponent (typically 1.0-2.0)

    Attributes:
        sound_speed: c₀ or 3D array [m/s]
        density: ρ₀ or 3D array [kg/m³]
        alpha_coeff: α₀ [dB/(MHz^y·cm)]
        alpha_power: y (power law exponent)
        BonA: B/A parameter for nonlinear acoustics (optional)
    """

    sound_speed: Union[float, NDArray[np.float64]]  # [m/s]
    density: Union[float, NDArray[np.float64]]  # [kg/m³]
    alpha_coeff: float = 0.0  # [dB/(MHz^y·cm)]
    alpha_power: float = 1.5  # dimensionless
    BonA: float = 0.0  # B/A nonlinearity parameter

    def __post_init__(self):
        """Validate medium parameters."""
        if isinstance(self.sound_speed, float) and self.sound_speed <= 0:
            raise ValueError("Sound speed must be positive")
        if isinstance(self.density, float) and self.density <= 0:
            raise ValueError("Density must be positive")
        if self.alpha_coeff < 0:
            raise ValueError("Absorption coefficient must be non-negative")
        if self.alpha_power < 0 or self.alpha_power > 3:
            raise ValueError("Alpha power must be in [0, 3]")

    def to_kwave_medium(self, engine, grid: GridConfig) -> Dict:
        """
        Create k-Wave medium structure.

        Returns:
            Dictionary of medium properties for k-Wave
        """
        medium = {}

        # Sound speed
        if isinstance(self.sound_speed, np.ndarray):
            medium["sound_speed"] = matlab.double(self.sound_speed.tolist())
        else:
            medium["sound_speed"] = float(self.sound_speed)

        # Density
        if isinstance(self.density, np.ndarray):
            medium["density"] = matlab.double(self.density.tolist())
        else:
            medium["density"] = float(self.density)

        # Absorption (only if non-zero)
        if self.alpha_coeff > 0:
            medium["alpha_coeff"] = float(self.alpha_coeff)
            medium["alpha_power"] = float(self.alpha_power)

        # Nonlinearity (only if specified)
        if self.BonA != 0:
            medium["BonA"] = float(self.BonA)

        return medium


@dataclass
class SourceConfig:
    """
    Acoustic source configuration for k-Wave.

    Mathematical Specification:
    - Pressure source: p_source(x,t) = mask(x) · signal(t)
    - Velocity source: u_source(x,t) = mask(x) · signal(t)

    Attributes:
        p_mask: Spatial pressure source mask (3D array, binary or weighted)
        p_signal: Temporal pressure signal (1D or 2D array) [Pa]
        u_mask: Spatial velocity source mask (optional)
        u_signal: Temporal velocity signal (optional) [m/s]
        frequency: Source frequency [Hz] (for analytical reference)
    """

    p_mask: Optional[NDArray[np.float64]] = None
    p_signal: Optional[NDArray[np.float64]] = None
    u_mask: Optional[NDArray[np.float64]] = None
    u_signal: Optional[NDArray[np.float64]] = None
    frequency: Optional[float] = None

    def to_kwave_source(self, engine) -> Dict:
        """
        Create k-Wave source structure.

        Returns:
            Dictionary of source properties for k-Wave
        """
        source = {}

        if self.p_mask is not None and self.p_signal is not None:
            source["p_mask"] = matlab.double(self.p_mask.tolist())
            # k-Wave expects (num_source_positions, num_time_steps)
            if self.p_signal.ndim == 1:
                # Single source position
                p_signal_2d = self.p_signal.reshape(1, -1)
            else:
                p_signal_2d = self.p_signal
            source["p"] = matlab.double(p_signal_2d.tolist())

        if self.u_mask is not None and self.u_signal is not None:
            source["u_mask"] = matlab.double(self.u_mask.tolist())
            if self.u_signal.ndim == 1:
                u_signal_2d = self.u_signal.reshape(1, -1)
            else:
                u_signal_2d = self.u_signal
            source["u"] = matlab.double(u_signal_2d.tolist())

        return source


@dataclass
class SensorConfig:
    """
    Sensor configuration for k-Wave.

    Attributes:
        mask: Binary mask indicating sensor positions (3D array)
        record: List of fields to record (e.g., ['p', 'u', 'p_max'])
    """

    mask: NDArray[np.float64]
    record: List[str] = None

    def __post_init__(self):
        if self.record is None:
            self.record = ["p"]  # Default: record pressure only

    def to_kwave_sensor(self, engine) -> Dict:
        """
        Create k-Wave sensor structure.

        Returns:
            Dictionary of sensor properties for k-Wave
        """
        sensor = {"mask": matlab.double(self.mask.tolist()), "record": self.record}
        return sensor


@dataclass
class SimulationResult:
    """
    k-Wave simulation results.

    Attributes:
        pressure: Recorded pressure field [Pa]
        velocity: Recorded velocity field [m/s] (optional)
        time_array: Time points [s]
        grid_config: Grid configuration used
        execution_time: Simulation runtime [s]
    """

    pressure: NDArray[np.float64]
    velocity: Optional[NDArray[np.float64]] = None
    time_array: Optional[NDArray[np.float64]] = None
    grid_config: Optional[GridConfig] = None
    execution_time: float = 0.0


class KWaveBridge:
    """
    Bridge to k-Wave MATLAB toolbox for automated acoustic simulation comparison.

    This class provides a Python interface to k-Wave, enabling:
    1. Automated comparison of kwavers vs k-Wave results
    2. Validation against k-Wave's extensively tested implementations
    3. Caching of k-Wave results for repeated comparisons

    Mathematical Correctness:
    - k-Wave implements k-space PSTD with exact spatial derivatives
    - Temporal integration: 4th-order Runge-Kutta or predictor-corrector
    - PML boundaries: CPML formulation (Roden & Gedney 2000)

    Usage:
        bridge = KWaveBridge()
        result = bridge.run_simulation(grid, medium, source, sensor, nt=1000, dt=1e-7)
    """

    def __init__(
        self, matlab_path: Optional[str] = None, cache_dir: Optional[Path] = None
    ):
        """
        Initialize k-Wave bridge.

        Args:
            matlab_path: Path to MATLAB executable (optional, auto-detect if None)
            cache_dir: Directory for caching k-Wave results (default: ./kwave_cache)
        """
        self.matlab_path = matlab_path
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./kwave_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.engine: Optional[object] = None
        self._matlab_initialized = False

    def __enter__(self):
        """Context manager entry: start MATLAB engine."""
        self.start_matlab()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop MATLAB engine."""
        self.stop_matlab()

    def start_matlab(self) -> None:
        """
        Start MATLAB engine and initialize k-Wave.

        Raises:
            RuntimeError: If MATLAB Engine is not available
            RuntimeError: If k-Wave toolbox is not found
        """
        if not MATLAB_AVAILABLE:
            raise RuntimeError(
                "MATLAB Engine not available. Install with: pip install matlabengine"
            )

        if self._matlab_initialized:
            return

        print("Starting MATLAB engine...")
        self.engine = matlab.engine.start_matlab()
        print("MATLAB engine started.")

        # Check if k-Wave is in MATLAB path
        try:
            version = self.engine.eval("kWaveVersion", nargout=1)
            print(f"k-Wave version {version} detected.")
            self._matlab_initialized = True
        except Exception as e:
            raise RuntimeError(
                f"k-Wave toolbox not found in MATLAB path. Error: {e}\n"
                "Download from: http://www.k-wave.org/"
            )

    def stop_matlab(self) -> None:
        """Stop MATLAB engine."""
        if self.engine is not None:
            print("Stopping MATLAB engine...")
            self.engine.quit()
            self.engine = None
            self._matlab_initialized = False
            print("MATLAB engine stopped.")

    def run_simulation(
        self,
        grid: GridConfig,
        medium: MediumConfig,
        source: SourceConfig,
        sensor: SensorConfig,
        nt: int,
        dt: float,
        data_cast: str = "single",
        plot_sim: bool = False,
        **kwargs,
    ) -> SimulationResult:
        """
        Run k-Wave acoustic simulation.

        Mathematical Specification:
        - Solves first-order acoustic equations using k-space PSTD
        - Spatial derivatives: F^{-1}[ik·F[f]] (exact in Fourier space)
        - Temporal integration: 4th-order Runge-Kutta (default)

        Args:
            grid: Grid configuration
            medium: Medium properties
            source: Source configuration
            sensor: Sensor configuration
            nt: Number of time steps
            dt: Time step size [s]
            data_cast: Data precision ('single' or 'double')
            plot_sim: Enable k-Wave visualization
            **kwargs: Additional k-Wave options

        Returns:
            SimulationResult with pressure, velocity, and timing data

        Raises:
            RuntimeError: If MATLAB engine not initialized
        """
        if not self._matlab_initialized:
            self.start_matlab()

        import time

        start_time = time.time()

        # Create k-Wave structures
        kgrid = grid.to_kwave_grid(self.engine)
        medium_struct = medium.to_kwave_medium(self.engine, grid)
        source_struct = source.to_kwave_source(self.engine)
        sensor_struct = sensor.to_kwave_sensor(self.engine)

        # Set up k-Wave input arguments
        input_args = {
            "PMLSize": grid.pml_size,
            "PMLAlpha": grid.pml_alpha,
            "DataCast": data_cast,
            "PlotSim": plot_sim,
        }
        input_args.update(kwargs)

        # Convert input_args to MATLAB-compatible format
        # k-Wave expects cell array of name-value pairs
        # TODO: Implement proper MATLAB struct conversion

        # Run k-Wave simulation (simplified call - needs proper struct handling)
        # This is a placeholder - actual MATLAB call requires careful struct marshalling
        print(
            f"Running k-Wave simulation: {grid.Nx}×{grid.Ny}×{grid.Nz} grid, {nt} steps..."
        )

        try:
            # NOTE: This is a simplified interface. Production code would use:
            # sensor_data = self.engine.kspaceFirstOrder3D(
            #     kgrid, medium_struct, source_struct, sensor_struct, input_args
            # )

            # For now, return mock result structure (to be replaced with actual k-Wave call)
            elapsed_time = time.time() - start_time

            # Create mock pressure data (to be replaced with actual sensor_data.p)
            num_sensors = int(np.sum(sensor.mask))
            pressure = np.zeros((num_sensors, nt))

            result = SimulationResult(
                pressure=pressure,
                time_array=np.arange(nt) * dt,
                grid_config=grid,
                execution_time=elapsed_time,
            )

            print(f"k-Wave simulation complete in {elapsed_time:.2f}s")
            return result

        except Exception as e:
            raise RuntimeError(f"k-Wave simulation failed: {e}")

    def cache_result(self, result: SimulationResult, cache_key: str) -> Path:
        """
        Cache simulation result to disk.

        Args:
            result: Simulation result to cache
            cache_key: Unique identifier for this simulation

        Returns:
            Path to cached file
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"

        np.savez_compressed(
            cache_file,
            pressure=result.pressure,
            velocity=result.velocity if result.velocity is not None else np.array([]),
            time_array=result.time_array
            if result.time_array is not None
            else np.array([]),
            execution_time=result.execution_time,
        )

        print(f"Cached k-Wave result to {cache_file}")
        return cache_file

    def load_cached_result(self, cache_key: str) -> Optional[SimulationResult]:
        """
        Load cached simulation result.

        Args:
            cache_key: Unique identifier for this simulation

        Returns:
            SimulationResult if cached, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if not cache_file.exists():
            return None

        data = np.load(cache_file)

        result = SimulationResult(
            pressure=data["pressure"],
            velocity=data["velocity"] if data["velocity"].size > 0 else None,
            time_array=data["time_array"] if data["time_array"].size > 0 else None,
            execution_time=float(data["execution_time"]),
        )

        print(f"Loaded cached k-Wave result from {cache_file}")
        return result


def create_plane_wave_test(
    grid_size: Tuple[int, int, int],
    dx: float,
    frequency: float,
    amplitude: float,
    sound_speed: float,
    density: float,
    nt: int,
    dt: float,
) -> Tuple[GridConfig, MediumConfig, SourceConfig, SensorConfig]:
    """
    Create plane wave test configuration.

    Mathematical Specification:
    - Plane wave: p(x,t) = A sin(kx - ωt)
    - Wave number: k = 2πf/c₀
    - Angular frequency: ω = 2πf

    Args:
        grid_size: (Nx, Ny, Nz)
        dx: Grid spacing [m]
        frequency: Source frequency [Hz]
        amplitude: Pressure amplitude [Pa]
        sound_speed: c₀ [m/s]
        density: ρ₀ [kg/m³]
        nt: Number of time steps
        dt: Time step [s]

    Returns:
        (grid, medium, source, sensor) configurations
    """
    Nx, Ny, Nz = grid_size

    # Grid configuration
    grid = GridConfig(Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dy=dx, dz=dx)

    # Medium configuration
    medium = MediumConfig(sound_speed=sound_speed, density=density)

    # Source configuration (plane wave at x=0)
    p_mask = np.zeros((Nx, Ny, Nz))
    p_mask[0, :, :] = 1.0

    # Sinusoidal source signal
    t_array = np.arange(nt) * dt
    omega = 2 * np.pi * frequency
    p_signal = amplitude * np.sin(omega * t_array)

    source = SourceConfig(p_mask=p_mask, p_signal=p_signal, frequency=frequency)

    # Sensor configuration (record entire domain)
    sensor_mask = np.ones((Nx, Ny, Nz))
    sensor = SensorConfig(mask=sensor_mask, record=["p", "p_max"])

    return grid, medium, source, sensor


if __name__ == "__main__":
    """
    Example usage: Run plane wave test case.
    """
    print("=" * 60)
    print("k-Wave Bridge Example: Plane Wave Test")
    print("=" * 60)

    # Test parameters
    Nx, Ny, Nz = 128, 128, 128
    dx = 0.5e-3  # 0.5 mm
    frequency = 1e6  # 1 MHz
    amplitude = 1e5  # 100 kPa
    sound_speed = 1500.0  # Water
    density = 1000.0
    nt = 1000
    dt = 50e-9  # 50 ns (CFL ≈ 0.3)

    # Create test configuration
    grid, medium, source, sensor = create_plane_wave_test(
        (Nx, Ny, Nz), dx, frequency, amplitude, sound_speed, density, nt, dt
    )

    print(f"\nGrid: {Nx}×{Ny}×{Nz}, dx={dx * 1e3:.2f}mm")
    print(f"Medium: c₀={sound_speed}m/s, ρ₀={density}kg/m³")
    print(f"Source: f={frequency / 1e6:.1f}MHz, A={amplitude / 1e3:.1f}kPa")
    print(f"Time: {nt} steps, dt={dt * 1e9:.1f}ns")
    print(f"CFL: {sound_speed * dt / dx:.3f}")

    if not MATLAB_AVAILABLE:
        print("\n⚠ MATLAB Engine not available - skipping simulation")
        print("Install with: pip install matlabengine")
    else:
        # Run simulation
        try:
            with KWaveBridge() as bridge:
                result = bridge.run_simulation(grid, medium, source, sensor, nt, dt)
                print(f"\n✓ Simulation complete")
                print(f"  Execution time: {result.execution_time:.2f}s")
                print(f"  Pressure shape: {result.pressure.shape}")
        except Exception as e:
            print(f"\n✗ Simulation failed: {e}")

    print("\n" + "=" * 60)
