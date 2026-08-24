# kwavers-python: Python Bindings for Kwavers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/built%20with-rust-orange.svg)](https://www.rust-lang.org/)

Python bindings for the [kwavers](https://github.com/ryancinsight/kwavers) ultrasound
simulation library, providing a k-Wave-compatible API for acoustic wave propagation
simulations.

## Overview

The `kwavers-python` distribution imports as `pykwavers` and brings the performance and
safety of Rust to Python-based acoustic simulations:

- 🚀 **High Performance**: Rust-backed numerical kernels with zero-copy numpy integration
- 🔒 **Memory Safe**: No segfaults, data races, or undefined behavior
- 🎯 **k-Wave Compatible**: Familiar API for easy comparison and migration
- 🧪 **Validated**: Direct comparison framework with k-Wave / k-wave-python
- 🌐 **Cross-Platform**: Windows, Linux, macOS stable-ABI (abi3-py38) wheels via PyO3

## Architecture

The bindings are a thin presentation layer. Type conversion, error mapping, and GIL
release live here; no domain logic, math, or state machines do.

```
┌─────────────────────────────────────┐
│   Python API (Presentation Layer)   │  ← pykwavers (this package)
├─────────────────────────────────────┤
│   Domain Layer (Rust layer crates)  │  ← kwavers-solver, -physics, -medium, …
├─────────────────────────────────────┤
│   Hardware Abstraction              │  ← CPU / GPU / SIMD
└─────────────────────────────────────┘
```

**Dependency direction**: Python → Rust, unidirectional. The Rust layer crates never
depend on `pyo3`.

## Installation

### From PyPI

```bash
pip install kwavers-python
```

### From Source (Development)

```bash
# Prerequisites: Rust toolchain (https://rustup.rs/)
pip install maturin

git clone https://github.com/ryancinsight/kwavers.git
cd kwavers

# Development install (editable)
maturin develop --release --manifest-path crates/kwavers-python/Cargo.toml

# Or build a wheel
maturin build --release --manifest-path crates/kwavers-python/Cargo.toml
pip install <target-dir>/wheels/kwavers_python-*.whl
```

### Optional Dependencies

```bash
# Plotting and comparison reports
pip install "kwavers-python[comparison]"

# MATLAB-free k-Wave Python comparison bridge (Python 3.10+)
pip install "kwavers-python[kwave]"

# MATLAB k-Wave bridge (requires MATLAB R2022b+)
pip install "kwavers-python[matlab]"

# For development
pip install "kwavers-python[dev]"
```

`import pykwavers` loads only the base API. Optional integration modules are loaded
through explicit submodule imports, such as `import pykwavers.comparison` or
`from pykwavers import kwave_bridge`; importing one without its declared extra propagates
the missing dependency error.

## Releases

GitHub Releases tagged `kwavers-python-v<version>` build one locked stable-ABI wheel per
operating system for Linux, Windows, and macOS. The workflow installs and imports each
wheel as `pykwavers`, verifies that its `kwavers-python` metadata version matches the
Cargo package version and release tag, attests and attaches the exact wheel set to the
GitHub Release, then publishes those same artifacts to PyPI through OIDC Trusted
Publishing.

## Quick Start

```python
import pykwavers as kw
import numpy as np

# Create computational grid (6.4×6.4×6.4 mm domain)
grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)

# Define acoustic medium (water at 20°C)
medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)

# Create plane wave source (1 MHz, 100 kPa)
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

# Create point sensor
sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))

# Run simulation
sim = kw.Simulation(grid, medium, source, sensor)
result = sim.run(time_steps=1000, dt=1e-8)

# Access results
print(f"Sensor data shape: {result.sensor_data.shape}")
print(f"Final time: {result.final_time*1e6:.2f} μs")
```

### MATLAB k-Wave Comparison

```python
from pykwavers.kwave_bridge import KWaveBridge, GridConfig, MediumConfig

grid_config = GridConfig(Nx=64, Ny=64, Nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
medium_config = MediumConfig(sound_speed=1500.0, density=1000.0)

with KWaveBridge() as bridge:
    result = bridge.run_simulation(grid_config, medium_config, source_config, sensor_config)
```

See [`examples/compare_plane_wave.py`](examples/compare_plane_wave.py) for the complete
comparison workflow.

## API Reference

### Grid

Computational domain with uniform Cartesian spacing.

```python
grid = kw.Grid(nx, ny, nz, dx, dy, dz)

grid.nx, grid.ny, grid.nz          # Grid dimensions
grid.dx, grid.dy, grid.dz          # Grid spacing [m]
grid.lx(), grid.ly(), grid.lz()    # Domain size [m]
grid.dimensions(), grid.spacing()  # As tuples
grid.total_points()                # Total grid points
```

**k-Wave equivalent**: `kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])`

### Medium

Acoustic material properties.

```python
# Homogeneous acoustic medium
medium = kw.Medium.homogeneous(
    sound_speed=1500.0,     # [m/s]
    density=1000.0,         # [kg/m³]
    absorption=0.0,         # [dB/(MHz^y·cm)] (optional)
    nonlinearity=0.0,       # B/A parameter (optional)
    alpha_power=1.0,        # power-law exponent y (optional)
    grid=None,              # (optional)
)

# Elastic media
medium = kw.Medium.elastic(c_compression, c_shear, density)
medium = kw.Medium.elastic_heterogeneous(...)
```

Invalid parameters raise `ValueError` at construction rather than producing an invalid
medium.

**k-Wave equivalent**: `medium.sound_speed`, `medium.density`

### Source

Acoustic wave excitation.

```python
source = kw.Source.plane_wave(grid=grid, frequency=1e6, amplitude=1e5)
source = kw.Source.point(position=(x, y, z), frequency=1e6, amplitude=1e5)
```

Transducer arrays (`kw.KWaveArray`, `kw.TransducerArray2D`, `kw.MultiRowRingArray`) are
accepted wherever a source is, individually or as a list.

**k-Wave equivalent**: `source.p_mask`, `source.p`

### Sensor

Field recording and sampling.

```python
sensor = kw.Sensor.point(position=(x, y, z))
sensor = kw.Sensor.grid()
```

**k-Wave equivalent**: `sensor.mask`

### Simulation

Main orchestrator for wave propagation.

```python
sim = kw.Simulation(grid, medium, source, sensor,
                    solver=kw.SolverType.PSTD,   # optional (see SolverType below)
                    pml_size=None)               # optional

result = sim.run(
    time_steps=1000,          # Number of time steps
    dt=1e-8,                  # [s] (optional; derived from the CFL condition when omitted)
    record_start_index=1,     # First step to record
    record_modes=None,        # e.g. ["p_max", "p_rms", "ux"]; defaults to the sensor's modes
)

result.sensor_data            # numpy array of sensor recordings [Pa]
result.sensor_data_shape      # (num_sensors, time_steps)
result.time                   # time vector [s]
result.time_steps, result.dt, result.final_time
result.p_max, result.p_min, result.p_rms   # None unless recorded
```

**k-Wave equivalent**: `sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor)`

### Beyond the k-Wave surface

The bindings also expose kwavers capabilities that have no k-Wave counterpart —
frequency-domain and elastic full-waveform inversion, transcranial FUS planning and
inversion from CT, theranostic workflows, passive acoustic mapping, and the analytical
cavitation/tissue kernels. `tests/test_bindings_surface.py` enumerates the canonical
public symbols.

## Mathematical Foundations

### Wave Equation

Linear acoustic wave equation in heterogeneous media:

```
∂²p/∂t² = c²(x)∇²p + source terms
```

### Discretization

`kw.SolverType` selects the scheme:

- **FDTD**: Finite-Difference Time-Domain (2nd/4th/6th order spatial accuracy)
- **PSTD**: Pseudospectral Time-Domain (spectral spatial accuracy)
- **Hybrid**: adaptive switching between FDTD and PSTD
- **PstdGpu**: GPU-resident PSTD; falls back to CPU PSTD when no adapter is present
- **Elastic**, **ElasticPSTD**: compressional + shear wave propagation
- **Helmholtz**, **BEM**: frequency-domain and boundary-element formulations
- **DG**: hybrid spectral / discontinuous-Galerkin for shock capturing

### Stability

CFL condition for explicit time-stepping:

```
dt ≤ CFL · dx / c_max,  where CFL = 1/√3 ≈ 0.577 (3D stability limit)
```

`Simulation.run` derives `dt` from this condition when it is not passed explicitly.

### Boundaries

- **PML**: Perfectly Matched Layers (Roden & Gedney 2000)
- **Periodic**: Phase-periodic boundaries for infinite media
- **Rigid**: Hard wall reflections

### Absorption

Power-law frequency-dependent absorption (Szabo 1994):

```
α(ω) = α₀ |ω|^y
```

where y ∈ [0, 3] (y = 2 for soft tissue).

## Comparison with k-Wave

### API Compatibility

| Feature | k-Wave (MATLAB) | k-wave-python | pykwavers |
|---------|-----------------|---------------|-----------|
| Grid creation | `kWaveGrid(...)` | `kWaveGrid(...)` | `Grid(...)` |
| Medium properties | `medium.sound_speed` | `medium.sound_speed` | `Medium.homogeneous(...)` |
| Source definition | `source.p_mask`, `source.p` | `source.p_mask`, `source.p` | `Source.plane_wave(...)` |
| Sensor mask | `sensor.mask` | `sensor.mask` | `Sensor.point(...)` |
| Simulation | `kspaceFirstOrder3D(...)` | `kspaceFirstOrder3D(...)` | `Simulation(...).run(...)` |

### Parity thresholds

Cross-implementation agreement is asserted, not assumed. The per-solver tolerance
profiles live in `pykwavers.comparison` and are enforced by `tests/test_solver_parity.py`:

| Solver | Relative L2 | L∞ | Correlation |
|---|---|---|---|
| FDTD | < 1.50 | < 2.00 | > 0.40 |
| PSTD | < 0.90 | < 1.20 | > 0.65 |

PSTD carries the stricter profile; the FDTD profile is the current tracked bound, not a
target. `pykwavers.comparison` also generates the comparison report consumed by
`validation_reports/`.

### Performance

Runtime comparisons are produced by the comparison harness on the machine under test
rather than quoted here — a speedup figure without its stored baseline, grid size, and
machine class is not evidence. Run
`pytest tests/ --benchmark-only` (with the `dev` extra installed) to measure locally.

## Examples

### Plane Wave Propagation

```bash
python examples/compare_plane_wave.py
```

Validates plane wave propagation against k-Wave and reports relative L2, L∞, and
correlation against the profile above.

### Point Source Radiation

```python
source = kw.Source.point(position=(0.0, 0.0, 0.0), frequency=1e6, amplitude=1e5)
sensor = kw.Sensor.grid()          # Record the entire field

result = sim.run(time_steps=1000)
# Verify 1/r geometric spreading: |p(r)| ∝ 1/r for r >> λ
```

## Development

### Building from Source

```bash
pip install maturin
maturin develop --release --manifest-path crates/kwavers-python/Cargo.toml
```

### Running Tests

```bash
cd crates/kwavers-python && pytest tests/ -v   # Python-side binding tests
cargo test -p kwavers-python                   # Rust-side tests
```

### Code Quality

```bash
black python/ examples/
ruff check python/ examples/
mypy python/

cargo fmt -p kwavers-python
cargo clippy -p kwavers-python --all-targets -- -D warnings
```

## Status

Alpha — the API may change. The k-Wave-compatible surface (grid, medium, source, sensor,
FDTD/PSTD/Hybrid/DG simulation with PML boundaries and recorded sensor modes) is
implemented and parity-tested. Heterogeneous and elastic media, transducer arrays, FWI,
and the transcranial/theranostic workflows are exposed and covered by the Python test
suite. GPU execution is selected at run time by the Rust `kwavers-gpu` backend when a
device is present.

## References

1. **k-Wave**: Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the
   simulation and reconstruction of photoacoustic wave fields." *Journal of Biomedical
   Optics*, 15(2), 021314.
2. **k-wave-python**: Jaros, J., et al. (2016). "Full-wave nonlinear ultrasound simulation
   on distributed clusters with applications in high-intensity focused ultrasound."
   *The International Journal of High Performance Computing Applications*, 30(2), 137–155.
3. **Absorption**: Szabo, T. L. (1994). "Time domain wave equations for lossy media
   obeying a frequency power law." *The Journal of the Acoustical Society of America*,
   96(1), 491–500.
4. **PML**: Roden, J. A., & Gedney, S. D. (2000). "Convolution PML (CPML): An efficient
   FDTD implementation of the CFS-PML for arbitrary media." *Microwave and Optical
   Technology Letters*, 27(5), 334–339.

## Contributing

Fork, branch, implement with tests and documentation, run the quality checks above, and
open a pull request. Workspace layout and design principles are in the
[repository README](https://github.com/ryancinsight/kwavers#readme).

## License

MIT — see the repository [LICENSE](https://github.com/ryancinsight/kwavers/blob/main/LICENSE).

## Contact

**Ryan Clanton PhD** · <ryanclanton@outlook.com> · [@ryancinsight](https://github.com/ryancinsight)
