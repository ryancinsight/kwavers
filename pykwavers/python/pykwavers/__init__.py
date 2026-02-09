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

# Import Rust extension module
# Import Python submodules for comparison and validation
from . import comparison, kwave_bridge, kwave_python_bridge
from ._pykwavers import (
    Grid,
    Medium,
    Sensor,
    Simulation,
    SimulationResult,
    SolverType,
    Source,
    __author__,
    __version__,
)

# Public API
__all__ = [
    # Core classes
    "Grid",
    "Medium",
    "Source",
    "Sensor",
    "Simulation",
    "SimulationResult",
    "SolverType",
    # Submodules
    "comparison",
    "kwave_python_bridge",
    "kwave_bridge",
    # Metadata
    "__version__",
    "__author__",
]

# Module-level metadata
__doc_format__ = "numpy"
__license__ = "MIT"
__copyright__ = "Copyright 2026 Ryan Clanton PhD"
