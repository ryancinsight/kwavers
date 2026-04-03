"""
pykwavers: Python Bindings for kwavers Ultrasound Simulation Library

A Rust-backed Python library for acoustic wave simulation with an API
compatible with k-Wave/k-wave-python for direct comparison and validation.

## Quick Start

```python
import pykwavers as kw
import numpy as np

# Create computational grid (similar to kWaveGrid)
grid = kw.Grid(nx=128, ny=128, nz=128, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)

# Define acoustic medium (similar to k-Wave medium struct)
medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)

# Create acoustic source
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

# Create sensor for field recording
sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))

# Run simulation
sim = kw.Simulation(grid, medium, source, sensor)
result = sim.run(time_steps=1000, dt=1e-8)

# Access results
print(f"Sensor data shape: {result.sensor_data.shape}")
print(f"Final time: {result.final_time:.2e} s")
```

## API Design Philosophy

The API mirrors k-Wave's structure for ease of comparison:
- **Grid**: Computational domain (equivalent to `kWaveGrid`)
- **Medium**: Acoustic properties (equivalent to k-Wave `medium` struct)
- **Source**: Wave excitation (equivalent to k-Wave `source` struct)
- **Sensor**: Field recording (equivalent to k-Wave `sensor` struct)
- **Simulation**: Main orchestrator (equivalent to `kspaceFirstOrder3D`)

## Architecture

Following Clean Architecture principles:
- **Presentation Layer**: Python API (this package)
- **Domain Layer**: Core kwavers library (Rust)
- **Dependency Direction**: Python → Rust (unidirectional)

## Mathematical Foundations

- **Wave Equation**: ∂²p/∂t² = c²∇²p + source terms
- **Discretization**: FDTD (2nd/4th/6th/8th order) or PSTD (spectral)
- **Stability**: CFL condition dt ≤ (dx/c_max)/√3
- **Boundaries**: PML (Perfectly Matched Layers)
- **Absorption**: Power-law α(ω) = α₀|ω|^y

## References

1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for simulation
   and reconstruction of photoacoustic wave fields." J. Biomed. Opt., 15(2).
2. kwavers architecture documentation
3. k-wave-python documentation

Author: Ryan Clanton PhD (@ryancinsight)
License: MIT
Repository: https://github.com/ryancinsight/kwavers
"""

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

# Import Rust extension module
_extension_override = os.getenv("PYKWAVERS_EXTENSION_PATH")
if _extension_override:
    _extension_path = Path(_extension_override).expanduser().resolve()
    _loader = importlib.machinery.ExtensionFileLoader(
        f"{__name__}._pykwavers",
        str(_extension_path),
    )
    _spec = importlib.util.spec_from_loader(
        f"{__name__}._pykwavers",
        _loader,
        origin=str(_extension_path),
    )
    if _spec is None:
        raise ImportError(f"Failed to create module spec for {_extension_path}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[f"{__name__}._pykwavers"] = _module
    _loader.exec_module(_module)

# Import Python submodules for comparison and validation
from . import comparison, kwave_bridge, kwave_python_bridge
from .parity_targets import PARITY_THRESHOLDS, evaluate_parity
from ._pykwavers import (
    # Core classes
    Grid,
    GpuPstdSession,
    Medium,
    Sensor,
    Simulation,
    SimulationResult,
    SolverType,
    Source,
    TransducerArray2D,
    # Phase 22: PID, Registration, Bubble Field
    PIDController,
    BubbleField,
    resample_to_target_grid,
    # Signal generation
    tone_burst,
    create_cw_signals,
    get_win,
    # Geometry
    make_disc,
    make_ball,
    make_sphere,
    make_circle,
    make_line,
    # Unit conversion
    db2neper,
    neper2db,
    freq2wavenumber,
    hounsfield2density,
    hounsfield2soundspeed,
    # Water properties (temperature-dependent)
    water_sound_speed,
    water_density,
    water_absorption,
    water_nonlinearity,
    # Signal processing
    add_noise,
    # Metadata
    __author__,
    __version__,
)

# ============================================================================
# Pure-Python k-Wave parity utilities
# ============================================================================
# These functions match the k-Wave/k-wave-python API and are implemented here
# in pure NumPy/SciPy — no Rust binding required since they are post-processing
# utilities that operate on NumPy arrays.

import numpy as _np


def gaussian(N: int, var: float, magnitude: float = 1.0) -> "_np.ndarray":
    """Create a Gaussian distribution with unit area.

    Matches k-Wave ``makeGaussian`` semantics.

    Parameters
    ----------
    N : int
        Number of samples (should be odd for symmetric result).
    var : float
        Variance (width²) of the Gaussian.  A larger value gives a wider pulse.
    magnitude : float, optional
        Peak amplitude.  Default is 1.0.

    Returns
    -------
    numpy.ndarray
        1-D array of length *N* containing the Gaussian samples.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``makeGaussian``.
    """
    t = _np.arange(-(N - 1) / 2.0, (N - 1) / 2.0 + 1)
    return magnitude * _np.exp(-(t ** 2) / (2.0 * var))


def spect(
    x: "_np.ndarray",
    fs: float,
    *,
    unwrap_phase: bool = False,
) -> "tuple[_np.ndarray, _np.ndarray, _np.ndarray]":
    """Single-sided amplitude spectrum via FFT.

    Matches k-Wave ``spect`` semantics.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal (1-D time series).
    fs : float
        Sampling frequency in Hz.
    unwrap_phase : bool, optional
        If True, unwrap the phase spectrum.  Default False.

    Returns
    -------
    f : numpy.ndarray
        Frequency axis in Hz (single-sided, 0 to fs/2).
    amp : numpy.ndarray
        Single-sided amplitude spectrum (peak amplitude, not RMS).
    phase : numpy.ndarray
        Phase spectrum in radians.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``spect``.
    """
    x = _np.asarray(x, dtype=float)
    n = len(x)
    fft_x = _np.fft.rfft(x)
    # Single-sided amplitude: double non-DC / non-Nyquist bins
    amp = _np.abs(fft_x) / n
    amp[1:-1] *= 2.0
    phase = _np.angle(fft_x)
    if unwrap_phase:
        phase = _np.unwrap(phase)
    f = _np.fft.rfftfreq(n, d=1.0 / fs)
    return f, amp, phase


def extract_amp_phase(
    x: "_np.ndarray",
    f: float,
    fs: float,
    *,
    dim: int = 0,
) -> "tuple[float, float]":
    """Extract amplitude and phase of a signal at a given frequency.

    Matches k-Wave ``extractAmpPhase`` semantics (scalar output for 1-D input).

    Parameters
    ----------
    x : numpy.ndarray
        Input signal (1-D time series or multi-dimensional array).
    f : float
        Target frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    dim : int, optional
        Axis along which to compute the FFT for multi-dimensional input.
        Default 0.

    Returns
    -------
    amp : float
        Peak amplitude at frequency *f*.
    phase : float
        Phase in radians at frequency *f*.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``extractAmpPhase``.
    """
    x = _np.asarray(x, dtype=float)
    n = x.shape[dim]
    freqs = _np.fft.rfftfreq(n, d=1.0 / fs)
    # Find nearest frequency bin
    idx = int(_np.argmin(_np.abs(freqs - f)))
    fft_x = _np.fft.rfft(x, axis=dim)
    # Extract element along the transform axis
    idx_tuple = [slice(None)] * fft_x.ndim
    idx_tuple[dim] = idx
    coeff = fft_x[tuple(idx_tuple)]
    amp = float(_np.abs(coeff)) * 2.0 / n
    phase = float(_np.angle(coeff))
    return amp, phase


def cart2grid(
    kgrid,
    cart_data: "_np.ndarray",
) -> "_np.ndarray":
    """Map Cartesian point data onto the nearest simulation grid points.

    Matches k-Wave ``cart2grid`` semantics.

    Parameters
    ----------
    kgrid : Grid
        kwavers ``Grid`` object defining the simulation domain.
    cart_data : numpy.ndarray
        Array of shape ``(3, N)`` (or ``(2, N)`` for 2-D) containing
        Cartesian (x, y, z) coordinates of the N points.

    Returns
    -------
    numpy.ndarray
        Boolean or integer mask array with shape ``(nx, ny, nz)`` where
        ``1`` marks the nearest grid voxel for each input point.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``cart2grid``.
    """
    cart_data = _np.asarray(cart_data, dtype=float)
    nx, ny, nz = kgrid.nx, kgrid.ny, kgrid.nz
    dx, dy, dz = kgrid.dx, kgrid.dy, kgrid.dz
    mask = _np.zeros((nx, ny, nz), dtype=_np.int8)

    # Grid origin is at the centre of the domain
    x_vec = ((_np.arange(nx) - nx / 2.0) * dx)
    y_vec = ((_np.arange(ny) - ny / 2.0) * dy)
    z_vec = ((_np.arange(nz) - nz / 2.0) * dz)

    for pt_idx in range(cart_data.shape[1]):
        xi = int(_np.argmin(_np.abs(x_vec - cart_data[0, pt_idx])))
        yi = int(_np.argmin(_np.abs(y_vec - cart_data[1, pt_idx])))
        zi = int(_np.argmin(_np.abs(z_vec - cart_data[2, pt_idx]))) if cart_data.shape[0] > 2 else 0
        mask[xi, yi, zi] = 1

    return mask


def grid2cart(
    kgrid,
    grid_data: "_np.ndarray",
    cart_data: "_np.ndarray",
) -> "_np.ndarray":
    """Extract grid field values at Cartesian point positions.

    Matches k-Wave ``grid2cart`` semantics.  Uses nearest-grid-point lookup.

    Parameters
    ----------
    kgrid : Grid
        kwavers ``Grid`` object defining the simulation domain.
    grid_data : numpy.ndarray
        3-D field array of shape ``(nx, ny, nz)``.
    cart_data : numpy.ndarray
        Array of shape ``(3, N)`` containing Cartesian (x, y, z) coordinates.

    Returns
    -------
    numpy.ndarray
        1-D array of length N containing the field value at each Cartesian point.

    References
    ----------
    Treeby & Cox (2010), k-Wave MATLAB toolbox, ``grid2cart``.
    """
    cart_data = _np.asarray(cart_data, dtype=float)
    grid_data = _np.asarray(grid_data, dtype=float)
    nx, ny, nz = kgrid.nx, kgrid.ny, kgrid.nz
    dx, dy, dz = kgrid.dx, kgrid.dy, kgrid.dz

    x_vec = (_np.arange(nx) - nx / 2.0) * dx
    y_vec = (_np.arange(ny) - ny / 2.0) * dy
    z_vec = (_np.arange(nz) - nz / 2.0) * dz

    n_pts = cart_data.shape[1]
    values = _np.zeros(n_pts)
    for pt_idx in range(n_pts):
        xi = int(_np.argmin(_np.abs(x_vec - cart_data[0, pt_idx])))
        yi = int(_np.argmin(_np.abs(y_vec - cart_data[1, pt_idx])))
        zi = int(_np.argmin(_np.abs(z_vec - cart_data[2, pt_idx]))) if cart_data.shape[0] > 2 else 0
        values[pt_idx] = grid_data[xi, yi, zi]

    return values


# Public API
__all__ = [
    # Core classes
    "Grid",
    "GpuPstdSession",
    "Medium",
    "Source",
    "TransducerArray2D",
    "Sensor",
    "Simulation",
    "SimulationResult",
    "SolverType",
    # Phase 22: PID, Registration, Bubble Field
    "PIDController",
    "BubbleField",
    "resample_to_target_grid",
    # Submodules
    "comparison",
    "kwave_python_bridge",
    "kwave_bridge",
    # Signal generation
    "tone_burst",
    "create_cw_signals",
    "get_win",
    # Geometry (matching k-Wave toolbox)
    "make_disc",
    "make_ball",
    "make_sphere",
    "make_circle",
    "make_line",
    # Unit conversion
    "db2neper",
    "neper2db",
    "freq2wavenumber",
    "hounsfield2density",
    "hounsfield2soundspeed",
    # Water properties (temperature-dependent, matching k-wave-python)
    "water_sound_speed",
    "water_density",
    "water_absorption",
    "water_nonlinearity",
    # Signal processing
    "add_noise",
    # k-Wave parity utilities (pure Python)
    "gaussian",
    "spect",
    "extract_amp_phase",
    "cart2grid",
    "grid2cart",
    # Metadata
    "__version__",
    "__author__",
]

# Module-level metadata
__doc_format__ = "numpy"
__license__ = "MIT"
__copyright__ = "Copyright 2026 Ryan Clanton PhD"
