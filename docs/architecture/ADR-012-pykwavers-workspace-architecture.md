# ADR-012: Workspace Architecture with PyO3 Python Bindings

**Status**: Accepted  
**Date**: 2026-02-04  
**Sprint**: 217 Session 9  
**Author**: Ryan Clanton (@ryancinsight)

## Context

kwavers requires Python bindings to enable:
1. Direct comparison with k-Wave/k-wave-python for validation
2. Integration with Python scientific computing ecosystem (NumPy, SciPy, Matplotlib)
3. Jupyter notebook workflows for research and education
4. API compatibility with k-Wave for easy migration

**Problem**: Where should Python bindings live architecturally?

### Options Considered

1. **Module within kwavers** (`kwavers::python`)
   - Single crate, simpler structure
   - Violates bounded context separation
   - PyO3 dependencies pollute core domain
   - Feature flag complexity

2. **Separate crate, no workspace** (`pykwavers/` sibling)
   - Clean separation
   - Harder dependency management
   - No shared profiles/lints

3. **Workspace with bounded contexts** (CHOSEN)
   - Clean Architecture enforcement
   - Bounded contexts as crate boundaries
   - Dependency inversion (Python → Rust)
   - Independent versioning/features

## Decision

**Adopt workspace structure with `pykwavers` as separate crate.**

### Structure

```
kwavers/                    # Git repository root
├── Cargo.toml              # Workspace manifest
│   └── [workspace]
│       ├── members = ["kwavers", "pykwavers", "xtask"]
├── kwavers/                # Core library crate (domain + application)
│   ├── Cargo.toml          # [package] name = "kwavers"
│   └── src/
├── pykwavers/              # Python bindings crate (presentation layer)
│   ├── Cargo.toml          # [package] name = "pykwavers"
│   ├── pyproject.toml      # Python packaging (maturin)
│   ├── src/
│   │   └── lib.rs          # PyO3 bindings
│   └── python/
│       └── pykwavers/      # Pure Python helpers
│           ├── __init__.py
│           └── kwave_bridge.py
└── xtask/                  # Build tooling (existing)
```

### Dependency Direction

```
pykwavers → kwavers → [core domain]
   ✅          ✅

kwavers ⇢ pykwavers  ❌ FORBIDDEN
```

**Unidirectional dependencies enforce Clean Architecture.**

## Rationale

### 1. Clean Architecture Compliance

From `ARCHITECTURE.md` 9-layer hierarchy:

- **Layer 9 (Infrastructure)**: API/I/O adapters ← **Python bindings here**
- **Layer 8 (Presentation)**: Clinical/User-facing APIs
- **Layer 7 (Application)**: High-level simulation orchestration
- ...
- **Layer 1 (Domain)**: Pure physics/math

Python bindings are Layer 9 infrastructure adapters. Workspace enforces this boundary at compile-time.

### 2. Bounded Context Separation (DDD)

- **Core Domain Context**: `kwavers` = pure acoustic simulation (Rust-only)
- **Python Integration Context**: `pykwavers` = Python ecosystem adapter (PyO3)

Prevents:
- PyO3 dependencies leaking into domain logic
- Python-specific concerns in core crate
- Circular dependencies
- Feature flag explosion

### 3. Single Responsibility Principle

- `kwavers`: Acoustic simulation (FDTD, PSTD, PML, etc.)
- `pykwavers`: Python interface (bindings, NumPy conversion, k-Wave compatibility)
- `xtask`: Build automation

Each crate has one reason to change.

### 4. Independent Versioning

- `kwavers`: v3.0.0 (stable Rust API)
- `pykwavers`: v0.1.0 (alpha Python API)

Python API can evolve independently without breaking Rust semver.

### 5. Feature Independence

```toml
# kwavers features
minimal, gpu, pinn, full

# pykwavers features
minimal, numpy, matplotlib, jupyter, kwave-bridge
```

Python features don't clutter core library.

## API Design: k-Wave Compatibility

Following k-Wave structure for direct comparison:

| Component | k-Wave (MATLAB) | pykwavers (Python) | kwavers (Rust) |
|-----------|----------------|-------------------|---------------|
| Grid | `kWaveGrid([Nx,Ny,Nz], [dx,dy,dz])` | `Grid(nx, ny, nz, dx, dy, dz)` | `Grid::new(...)` |
| Medium | `medium.sound_speed = 1500` | `Medium.homogeneous(1500, 1000)` | `HomogeneousMedium::new(...)` |
| Source | `source.p_mask`, `source.p` | `Source.plane_wave(...)` | `GridSource::new(...)` |
| Sensor | `sensor.mask = mask_array` | `Sensor.point((x,y,z))` | `GridSensorSet::new(...)` |
| Simulation | `kspaceFirstOrder3D(...)` | `Simulation(...).run(...)` | `CoreSimulation::run(...)` |

**Design Goal**: Python users can translate k-Wave scripts to pykwavers with minimal changes.

## Implementation

### PyO3 Bindings (`pykwavers/src/lib.rs`)

```rust
#[pyclass]
pub struct Grid {
    inner: kwavers::domain::grid::Grid,
}

#[pymethods]
impl Grid {
    #[new]
    fn new(nx: usize, ny: usize, nz: usize, 
           dx: f64, dy: f64, dz: f64) -> PyResult<Self> {
        let inner = kwavers::Grid::new(nx, ny, nz, dx, dy, dz)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(Grid { inner })
    }
    
    #[getter]
    fn nx(&self) -> usize { self.inner.nx }
    
    fn total_points(&self) -> usize { self.inner.size() }
}
```

### Python Package (`pykwavers/python/pykwavers/__init__.py`)

```python
from ._pykwavers import Grid, Medium, Source, Sensor, Simulation

__all__ = ["Grid", "Medium", "Source", "Sensor", "Simulation"]
```

### Build System (`pykwavers/pyproject.toml`)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
python-source = "python"
module-name = "pykwavers._pykwavers"
features = ["minimal"]
```

## Consequences

### Positive

1. **Architectural Purity**
   - Clean separation of concerns
   - Enforced dependency direction
   - No circular dependencies

2. **Independent Evolution**
   - Python API can iterate quickly (v0.x)
   - Core Rust API remains stable (v3.x)
   - Breaking changes isolated

3. **Build Performance**
   - `cargo build -p kwavers` doesn't rebuild PyO3
   - `maturin develop -m pykwavers/Cargo.toml` only rebuilds bindings
   - Parallel compilation possible

4. **Testing Isolation**
   - Rust tests: `cargo test -p kwavers`
   - Python tests: `pytest pykwavers/tests/`
   - Comparison tests: `python pykwavers/examples/compare_plane_wave.py`

5. **Future Extensibility**
   - Add C API crate (`kwavers-c/`)
   - Add WASM bindings (`kwavers-wasm/`)
   - Add other language bindings without core changes

### Negative

1. **Workspace Complexity**
   - More Cargo.toml files to manage
   - Workspace-level profile warnings
   - Learning curve for contributors

2. **Path Dependencies**
   - `kwavers = { path = ".." }` must be maintained
   - CI must handle workspace builds

3. **Documentation Split**
   - Rust docs: `docs.rs/kwavers`
   - Python docs: Sphinx/MkDocs (future)

### Neutral

1. **Naming Convention**
   - Workspace: `kwavers` (Git repo)
   - Core crate: `kwavers` (Cargo package)
   - Python crate: `pykwavers` (follows `py-polars`, `pyo3` convention)

## Validation

### Compilation

```bash
# Workspace builds correctly
cargo build --workspace

# Individual crates compile
cargo build -p kwavers
cargo build -p pykwavers
cargo build -p xtask

# Python wheel builds
cd pykwavers
maturin build --release
```

### API Compatibility

Example: Plane wave propagation

**k-Wave (MATLAB)**:
```matlab
kgrid = kWaveGrid([64, 64, 64], [0.1e-3, 0.1e-3, 0.1e-3]);
medium.sound_speed = 1500;
medium.density = 1000;
source.p_mask = zeros(Nx, Ny, Nz);
source.p_mask(:, :, 1) = 1;
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor);
```

**pykwavers (Python)**:
```python
grid = kw.Grid(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3)
medium = kw.Medium.homogeneous(1500, 1000)
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
sensor = kw.Sensor.point((0.01, 0.01, 0.01))
sim = kw.Simulation(grid, medium, source, sensor)
result = sim.run(time_steps=1000, dt=1e-8)
```

**Similarity**: Structure and workflow nearly identical.

## Comparison Framework

### k-Wave Bridge (`pykwavers/python/pykwavers/kwave_bridge.py`)

Provides automated comparison with k-Wave via MATLAB Engine:

```python
from pykwavers.kwave_bridge import KWaveBridge, GridConfig

# Run identical simulation in k-Wave
with KWaveBridge() as bridge:
    kwave_result = bridge.run_simulation(grid_config, medium_config, ...)

# Compare with pykwavers
l2_error = np.linalg.norm(pykwavers_result - kwave_result) / np.linalg.norm(kwave_result)
```

### Acceptance Criteria (Sprint 217)

- L2 error < 0.01 (1% relative)
- L∞ error < 0.05 (5% maximum deviation)
- Phase error < 0.1 rad
- Runtime: pykwavers ≤ 5× k-Wave (goal: 2-3×)

## References

### Architecture

1. **Clean Architecture**: Martin, R. C. (2017). *Clean Architecture*. Prentice Hall.
2. **Domain-Driven Design**: Evans, E. (2003). *Domain-Driven Design*. Addison-Wesley.
3. **kwavers ARCHITECTURE.md**: 9-layer hierarchy, bounded contexts, SSOT principles.

### k-Wave

1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for simulation and reconstruction of photoacoustic wave fields." *J. Biomed. Opt.*, 15(2), 021314.
2. k-wave-python documentation: https://github.com/waltsims/k-wave-python

### PyO3

1. PyO3 User Guide: https://pyo3.rs/
2. Maturin Documentation: https://www.maturin.rs/

### Related ADRs

- **ADR-001**: Algebraic Architecture (typestates, trait-driven APIs)
- **ADR-010**: Performance Benchmarking Strategy
- **ADR-011**: Minimalist Production Architecture

## Alternatives Considered and Rejected

### Alternative 1: Python Wrapper Around C API

**Rejected**: Adds extra layer (Rust → C → Python). PyO3 provides direct Rust ↔ Python interop with zero-copy NumPy arrays.

### Alternative 2: Standalone pykwavers Repository

**Rejected**: Creates synchronization burden. Changes to core API require coordinated releases across repos.

### Alternative 3: Inline `#[cfg(feature = "python")]`

**Rejected**: Violates bounded context separation. PyO3 dependencies leak into core. Feature flag explosion.

## Migration Path (Future)

If workspace proves problematic:

1. **Option A**: Move pykwavers to separate repo
   - Use published `kwavers` crate from crates.io
   - Independent release cadence

2. **Option B**: Merge into single crate
   - Use feature flag `python = ["dep:pyo3"]`
   - Acceptable only if complexity is low

**Current Decision**: Workspace is optimal. No migration planned.

## Review and Approval

- **Proposed**: 2026-02-04
- **Reviewed**: Self (mathematical verification, architectural soundness)
- **Approved**: 2026-02-04 (Sprint 217 Session 9 completion)
- **Supersedes**: None (new architecture decision)

---

**Next Steps**:
1. ✅ Workspace structure implemented
2. ✅ PyO3 bindings skeleton complete
3. ✅ k-Wave bridge migrated
4. ✅ Example comparison script created
5. 🔄 Complete FDTD/PSTD backend integration (Sprint 217 Session 10)
6. 🔄 Implement NumPy array returns (resolve numpy 0.20 API)
7. 🔄 Run plane wave validation against k-Wave
8. 🔄 Benchmark performance vs k-Wave/jwave

**Status**: Foundation complete. API functional. Backend integration pending.