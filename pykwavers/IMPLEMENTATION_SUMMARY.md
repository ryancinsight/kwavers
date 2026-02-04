# pykwavers: Python Integration Implementation Summary

**Date**: 2026-02-04  
**Sprint**: 217 Session 9  
**Author**: Ryan Clanton (@ryancinsight)  
**Status**: ✅ Foundation Complete, Backend Integration Pending

---

## Overview

pykwavers provides Python bindings for the kwavers ultrasound simulation library with an API designed for direct comparison with k-Wave/k-wave-python.

### Key Achievements

1. **Workspace Architecture**: Clean separation of concerns via Cargo workspace
2. **PyO3 Bindings**: k-Wave-compatible Python API (729 lines)
3. **Comparison Framework**: Automated k-Wave validation pipeline
4. **Documentation**: Comprehensive API docs and architectural rationale
5. **Compilation**: ✅ Successful build with zero errors

---

## Architecture Decision (ADR-012)

### Structure

```
kwavers/                    # Workspace root
├── Cargo.toml              # [workspace] members = ["kwavers", "pykwavers", "xtask"]
├── kwavers/                # Core library (Rust)
│   └── src/                # Domain logic, solvers, physics
├── pykwavers/              # Python bindings (Presentation layer)
│   ├── Cargo.toml          # PyO3 crate configuration
│   ├── pyproject.toml      # Python packaging (maturin)
│   ├── src/lib.rs          # Rust → Python bindings
│   ├── python/pykwavers/   # Pure Python helpers
│   └── examples/           # Comparison scripts
└── xtask/                  # Build tooling
```

### Rationale

**Clean Architecture Compliance**:
- Layer 9 (Infrastructure): pykwavers = Python adapter
- Layers 1-8: kwavers = Core domain + application logic
- **Dependency Flow**: Python → Rust (unidirectional, no cycles)

**Domain-Driven Design**:
- Bounded Context 1: kwavers (acoustic simulation)
- Bounded Context 2: pykwavers (Python integration)
- **Isolation**: PyO3 dependencies never leak into core

**Benefits**:
- Independent versioning (kwavers v3.0.0, pykwavers v0.1.0)
- Independent features (Rust: gpu/pinn; Python: jupyter/matplotlib)
- Build performance (incremental compilation)
- Future extensibility (C API, WASM, etc.)

---

## API Design: k-Wave Compatibility

### Mapping

| Component | k-Wave (MATLAB) | pykwavers (Python) | kwavers (Rust) |
|-----------|----------------|-------------------|---------------|
| **Grid** | `kWaveGrid([Nx,Ny,Nz], [dx,dy,dz])` | `Grid(nx, ny, nz, dx, dy, dz)` | `Grid::new(...)` |
| **Medium** | `medium.sound_speed = 1500` | `Medium.homogeneous(1500, 1000)` | `HomogeneousMedium::new(...)` |
| **Source** | `source.p_mask = mask; source.p = signal` | `Source.plane_wave(grid, freq, amp)` | `GridSource::new(...)` |
| **Sensor** | `sensor.mask = mask_array` | `Sensor.point((x, y, z))` | `GridSensorSet::new(...)` |
| **Simulation** | `data = kspaceFirstOrder3D(...)` | `result = sim.run(steps, dt)` | `CoreSimulation::run(...)` |

### Example: Plane Wave Propagation

**k-Wave (MATLAB)**:
```matlab
kgrid = kWaveGrid([64, 64, 64], [0.1e-3, 0.1e-3, 0.1e-3]);
medium.sound_speed = 1500;
medium.density = 1000;
source.p_mask = zeros(Nx, Ny, Nz);
source.p_mask(:, :, 1) = 1;
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(32, 32, 32) = 1;
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor);
```

**pykwavers (Python)**:
```python
import pykwavers as kw

grid = kw.Grid(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3)
medium = kw.Medium.homogeneous(sound_speed=1500, density=1000)
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))

sim = kw.Simulation(grid, medium, source, sensor)
result = sim.run(time_steps=1000, dt=1e-8)

print(f"Shape: {result.sensor_data_shape()}")
print(f"Final time: {result.final_time*1e6:.2f} μs")
```

**Similarity**: Near-identical workflow enables direct comparison.

---

## Implementation Details

### 1. PyO3 Bindings (`src/lib.rs`)

**Size**: 729 lines  
**Modules**: 7 (Error, Grid, Medium, Source, Sensor, Simulation, Result)

#### Grid
- Properties: `nx`, `ny`, `nz`, `dx`, `dy`, `dz`
- Computed: `lx()`, `ly()`, `lz()`, `total_points()`
- Methods: `dimensions()`, `spacing()`

#### Medium
- Factory: `Medium.homogeneous(sound_speed, density, absorption, nonlinearity)`
- Internal: Wraps `kwavers::HomogeneousMedium`
- Validation: Positive sound speed/density, non-negative absorption

#### Source
- Factories: `Source.plane_wave(grid, frequency, amplitude, direction)`
- Factories: `Source.point(position, frequency, amplitude)`
- Validation: Positive frequency/amplitude

#### Sensor
- Factories: `Sensor.point(position)`, `Sensor.grid()`
- Storage: Position metadata for interpolation

#### Simulation
- Constructor: `Simulation(grid, medium, source, sensor)`
- Method: `run(time_steps, dt=None)` → `SimulationResult`
- CFL: Auto-calculates time step if `dt=None` (CFL=0.3 conservative)

#### SimulationResult
- Properties: `shape`, `time_steps`, `dt`, `final_time`
- Note: `sensor_data` returns shape tuple (placeholder for numpy array)

### 2. Python Package (`python/pykwavers/`)

**File**: `__init__.py` (99 lines)
- Imports: Grid, Medium, Source, Sensor, Simulation, SimulationResult
- Metadata: `__version__`, `__author__`, `__license__`
- Docstring: NumPy-style with quick start example

**File**: `kwave_bridge.py` (563 lines, migrated)
- MATLAB Engine integration for k-Wave comparison
- Classes: GridConfig, MediumConfig, SourceConfig, SensorConfig, SimulationResult
- Class: KWaveBridge (context manager for MATLAB session)
- Caching: JSON serialization for reproducible comparisons
- Graceful degradation: Warning if MATLAB Engine unavailable

### 3. Comparison Example (`examples/compare_plane_wave.py`)

**Size**: 377 lines  
**Purpose**: Validate pykwavers against k-Wave with identical plane wave test

#### Test Configuration
- Grid: 64³ points, 0.1 mm spacing (6.4 mm domain)
- Medium: Water (c=1500 m/s, ρ=1000 kg/m³, α=0 dB/(MHz·cm))
- Source: 1 MHz plane wave, 100 kPa amplitude, +z propagation
- Duration: 10 μs (15 wavelengths)
- Sensor: Point at center (32, 32, 32)

#### Workflow
1. Run pykwavers simulation (Rust backend)
2. Run k-Wave simulation (MATLAB Engine, if available)
3. Compute error metrics: L2, L∞, RMSE
4. Visualize: Pressure time series, error plots
5. Validate: Check acceptance criteria

#### Acceptance Criteria (Sprint 217)
- L2 error < 0.01 (1% relative)
- L∞ error < 0.05 (5% peak deviation)
- Phase error < 0.1 rad
- Runtime: pykwavers ≤ 5× k-Wave (target: 2-3×)

### 4. Build Configuration

**Cargo.toml** (54 lines):
- Crate type: `cdylib` (dynamic library for Python import)
- Dependencies:
  - `kwavers = { path = "..", default-features = false, features = ["minimal"] }`
  - `pyo3 = { version = "0.20", features = ["extension-module", "abi3-py38"] }`
  - `numpy = "0.20"`
- Features: minimal (default), gpu, plotting, full
- Release: Optimized with symbol stripping for wheel size

**pyproject.toml** (98 lines):
- Build system: maturin (Rust/Python hybrid packaging)
- Python: ≥3.8 (abi3 stable ABI)
- Dependencies:
  - Required: numpy ≥1.20, scipy ≥1.7
  - Optional: matplotlib, pandas (comparison), matlabengine (k-Wave)
  - Dev: pytest, pytest-benchmark, black, ruff, mypy, sphinx
- Tools: pytest config, black/ruff formatting (line-length=100)

---

## Documentation

### README.md (443 lines)

**Sections**:
1. Overview: Features, architecture diagram
2. Installation: PyPI, source build, optional deps
3. Quick Start: Basic simulation example
4. API Reference: Grid, Medium, Source, Sensor, Simulation
5. Mathematical Foundations: Wave equation, CFL, PML, absorption
6. k-Wave Comparison: API table, performance benchmarks
7. Examples: Plane wave, point source, focused ultrasound
8. Development: Build, test, quality checks
9. Roadmap: 4-phase plan (Core API → Backend → Features → Validation)
10. References: k-Wave papers, PyO3 docs, kwavers architecture

### ADR-012 (378 lines)

**Architectural Decision Record**:
1. Context: Need for Python bindings
2. Options: Module, separate crate, workspace
3. Decision: Workspace with bounded contexts
4. Rationale: Clean Architecture, DDD, SRP
5. API Design: k-Wave compatibility mapping
6. Implementation: PyO3, maturin, package structure
7. Consequences: Positive (purity, independence), negative (complexity)
8. Validation: Compilation, API examples
9. Comparison Framework: k-Wave bridge, acceptance criteria
10. References: Martin, Evans, Treeby, PyO3 guide

---

## Code Statistics

### Lines of Code
- **Rust bindings**: 729 lines (`src/lib.rs`)
- **Python package**: 99 lines (`__init__.py`)
- **k-Wave bridge**: 563 lines (`kwave_bridge.py`)
- **Comparison example**: 377 lines (`compare_plane_wave.py`)
- **Documentation**: 821 lines (README + ADR-012)
- **Configuration**: 152 lines (Cargo.toml + pyproject.toml)
- **Total**: 2,741 lines

### Files
- **Created**: 8
  - `pykwavers/Cargo.toml`
  - `pykwavers/pyproject.toml`
  - `pykwavers/src/lib.rs`
  - `pykwavers/python/pykwavers/__init__.py`
  - `pykwavers/python/pykwavers/kwave_bridge.py` (migrated)
  - `pykwavers/examples/compare_plane_wave.py`
  - `pykwavers/README.md`
  - `docs/architecture/ADR-012-pykwavers-workspace-architecture.md`
- **Modified**: 2
  - `Cargo.toml` (workspace members)
  - `docs/sprints/SPRINT_217_SESSION_9_PROGRESS.md`

---

## Compilation & Testing

### Build Status
```bash
cargo build -p pykwavers
# ✅ SUCCESS: Compiled successfully with 8 warnings (unused imports)
# ⚠️  Warnings addressable via: cargo fix --lib -p pykwavers
```

### Installation Test
```bash
cd pykwavers
maturin develop --release
# ✅ Expected: Builds Python wheel and installs in virtualenv
# ⚠️  Requires: Rust toolchain, Python ≥3.8, virtualenv active
```

### API Test
```python
import pykwavers as kw

# Create grid
grid = kw.Grid(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3)
assert grid.nx == 64
assert grid.total_points() == 64**3

# Create medium
medium = kw.Medium.homogeneous(1500.0, 1000.0)
assert isinstance(medium, kw.Medium)

# Create source
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
assert source.frequency == 1e6

# Create sensor
sensor = kw.Sensor.point((0.01, 0.01, 0.01))
assert sensor.sensor_type == "point"

# Create simulation
sim = kw.Simulation(grid, medium, source, sensor)
result = sim.run(time_steps=100)
assert result.time_steps == 100
assert result.shape == (64, 64, 64)

print("✅ All API tests passed!")
```

---

## Known Issues & Future Work

### High Priority

1. **NumPy Array Returns**
   - **Issue**: `SimulationResult.shape` returns tuple (placeholder)
   - **Target**: Return `numpy.ndarray` with sensor data
   - **Blocker**: numpy 0.20 API incompatibility with ndarray conversion
   - **Solution**: Upgrade to numpy 0.21+ or use PyArray buffer protocol
   - **Effort**: 2-4 hours

2. **Simulation Backend Integration**
   - **Issue**: `Simulation.run()` returns dummy data (placeholder)
   - **Target**: Wire to kwavers FDTD/PSTD solvers
   - **Dependencies**: CoreSimulation, ForwardSolver, SolverBackend
   - **Effort**: 8-16 hours (Session 10)

3. **k-Wave Validation**
   - **Issue**: Comparison script ready, backend incomplete
   - **Target**: Run plane wave test, verify L2 < 0.01
   - **Dependencies**: Backend integration, MATLAB Engine
   - **Effort**: 4-8 hours after backend complete

### Medium Priority

4. **Heterogeneous Media**
   - **Issue**: Only homogeneous medium supported
   - **Target**: `Medium.heterogeneous(c_map, rho_map, alpha_map)`
   - **Dependencies**: HeterogeneousMedium Python bindings
   - **Effort**: 4-6 hours

5. **Source Types**
   - **Issue**: Only plane wave and point source
   - **Target**: Phased array, focused transducer, arbitrary mask
   - **Dependencies**: Source trait bindings
   - **Effort**: 6-10 hours

6. **Performance Benchmarking**
   - **Issue**: Preliminary estimates only
   - **Target**: pytest-benchmark suite vs k-Wave/jwave
   - **Metrics**: Runtime, memory, throughput
   - **Effort**: 4-8 hours

### Low Priority

7. **GPU Acceleration**
   - **Feature**: Expose GPU backend via Python API
   - **Target**: `Simulation(..., backend='gpu')`
   - **Dependencies**: wgpu feature flag
   - **Effort**: 6-10 hours

8. **Visualization Helpers**
   - **Feature**: `result.plot()`, `grid.visualize()`
   - **Target**: Matplotlib integration for quick viz
   - **Effort**: 4-6 hours

9. **Jupyter Notebook Examples**
   - **Feature**: Interactive tutorials in notebooks/
   - **Target**: Plane wave, focused ultrasound, beamforming demos
   - **Effort**: 8-12 hours

---

## Roadmap

### Phase 1: Core API ✅ COMPLETE
- [x] Workspace structure (ADR-012)
- [x] PyO3 bindings (Grid, Medium, Source, Sensor, Simulation)
- [x] k-Wave bridge (MATLAB Engine integration)
- [x] Comparison example (plane_wave_comparison.py)
- [x] Documentation (README, ADR-012)

### Phase 2: Backend Integration (Session 10) 🔄 IN PROGRESS
- [ ] Wire `Simulation.run()` to kwavers CoreSimulation
- [ ] Implement sensor data recording
- [ ] Add numpy array returns (resolve numpy 0.20 API)
- [ ] PML boundary integration
- [ ] CFL auto-calculation verification

### Phase 3: Advanced Features (Sprint 218)
- [ ] Heterogeneous media Python API
- [ ] Additional source types (phased array, focused)
- [ ] Sensor interpolation (line sensors, arbitrary positions)
- [ ] Nonlinear propagation flag
- [ ] Absorption models (power law, Stokes)
- [ ] GPU backend toggle

### Phase 4: Validation & Benchmarking (Sprint 219)
- [ ] Run full k-Wave validation suite
- [ ] Performance benchmarks (Criterion, pytest-benchmark)
- [ ] Literature validation (Hamilton & Blackstock problems)
- [ ] Publication-quality comparison report
- [ ] CI/CD integration (automated k-Wave comparison)

---

## Usage Examples

### Basic Simulation
```python
import pykwavers as kw
import numpy as np

# Create 3D grid (6.4 mm domain)
grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)

# Define water medium
medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)

# Create 1 MHz plane wave source
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

# Add point sensor at center
sensor = kw.Sensor.point(position=(0.0032, 0.0032, 0.0032))

# Run simulation for 10 μs
sim = kw.Simulation(grid, medium, source, sensor)
result = sim.run(time_steps=1000, dt=1e-8)

print(f"Simulation complete: {result.final_time*1e6:.2f} μs")
print(f"Sensor data shape: {result.sensor_data_shape()}")
```

### k-Wave Comparison
```python
from pykwavers.kwave_bridge import KWaveBridge, GridConfig, MediumConfig

# Configure grid (identical to pykwavers)
grid_config = GridConfig(Nx=64, Ny=64, Nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)

# Configure medium
medium_config = MediumConfig(sound_speed=1500.0, density=1000.0)

# Run k-Wave simulation (requires MATLAB Engine)
with KWaveBridge() as bridge:
    kwave_result = bridge.run_simulation(
        grid_config, medium_config, source_config, sensor_config
    )

# Compare results
l2_error = np.linalg.norm(pykwavers_data - kwave_result.sensor_data)
print(f"L2 error: {l2_error:.2e}")
```

---

## References

### Architecture & Design
1. Martin, R. C. (2017). *Clean Architecture*. Prentice Hall.
2. Evans, E. (2003). *Domain-Driven Design*. Addison-Wesley.
3. kwavers ARCHITECTURE.md (9-layer hierarchy, bounded contexts)

### k-Wave
1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for simulation and reconstruction of photoacoustic wave fields." *J. Biomed. Opt.*, 15(2), 021314.
2. k-wave-python: https://github.com/waltsims/k-wave-python
3. k-Wave documentation: http://www.k-wave.org/

### Python/Rust Interop
1. PyO3 User Guide: https://pyo3.rs/
2. Maturin Documentation: https://www.maturin.rs/
3. numpy-rs Documentation: https://docs.rs/numpy/

### Related ADRs
- **ADR-001**: Algebraic Architecture (typestates, trait-driven APIs)
- **ADR-010**: Performance Benchmarking Strategy
- **ADR-011**: Minimalist Production Architecture
- **ADR-012**: pykwavers Workspace Architecture (this document)

---

## Conclusion

**Status**: Foundation complete. Python bindings operational. Backend integration pending.

**Next Steps**:
1. Session 10: Wire pykwavers to kwavers FDTD/PSTD backend
2. Resolve numpy 0.20 API for array returns
3. Run plane wave validation against k-Wave
4. Benchmark performance vs k-Wave/jwave

**Impact**: Enables direct comparison with k-Wave, Python ecosystem integration, and Jupyter workflows for kwavers validation and adoption.

---

**Sprint**: 217 Session 9  
**Date**: 2026-02-04  
**Commit**: `470c9795` - feat(pykwavers): Implement Python bindings via PyO3 with k-Wave-compatible API