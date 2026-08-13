# Kwavers 🌀

[![Repository](https://img.shields.io/badge/repo-ryancinsight%2Fkwavers-blue.svg)](https://github.com/ryancinsight/kwavers)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://docs.rs/kwavers)
[![Rust](https://img.shields.io/badge/rust-2021+-orange.svg)](https://www.rust-lang.org/)

**An interdisciplinary ultrasound-light physics simulation library.** Kwavers models acoustic wave propagation, cavitation dynamics, and sonoluminescence for multi-modal imaging research and physics studies.

![Theranostic feedback loop](docs/book/figures/theranostics_feedback_loop.svg)

## 📋 Library Components

### Physics Models
- **Acoustic Wave Propagation**: Linear and nonlinear wave equations
- **Cavitation Dynamics**: Bubble physics implementations
- **Multi-Physics Coupling**: Basic acoustic-thermal interactions
- **Electromagnetic Models**: Wave propagation in various media

### Numerical Methods
- **FDTD Solver**: Finite difference time domain implementation
- **PSTD Solver**: Pseudospectral time domain method
- **PINN Support**: Physics-informed neural networks (experimental)
- **Boundary Conditions**: Various absorbing and reflecting boundaries

### Application Areas
- **Research Simulations**: Acoustic wave propagation studies
- **Imaging Algorithms**: Basic beamforming and reconstruction
- **Material Modeling**: Acoustic properties of different media
- **Signal Processing**: Filtering and analysis utilities

## 📊 Current Development Status

**Recently completed:** the workspace crate split (ADR-011) — the ~460k-LOC
`kwavers` monolith is decomposed into per-layer crates to cut incremental build
times. There is **no facade**: consumers (including the Python bindings) depend on
the layer crates directly (`kwavers_core`, `kwavers_grid`, `kwavers_solver`, …).
The `kwavers` crate is now only a thin top-level **app/integration** crate — it
hosts the binary and the cross-cutting tests/examples/benches, and re-exports
nothing.

### Workspace layout

Versions are per crate, not workspace-wide; each crate's `Cargo.toml` is
authoritative and the table mirrors it. Crates are listed in dependency order:
every crate depends only on crates above it.

| Crate | Version | Layer / responsibility |
|-------|---------|------------------------|
| `kwavers-core` | 3.0.0 | Constants, error types, arena allocation, time/logging utilities |
| `kwavers-alloc-probe` | 0.1.0 | Thread-scoped allocation counting for allocation-contract tests |
| `kwavers-math` | 3.0.0 | FFT, linear algebra, numerics, geometry, statistics, SIMD |
| `kwavers-grid` | 3.0.0 | Cartesian/cylindrical grids, coordinates, topology, operators, k-space utilities |
| `kwavers-field` | 3.0.0 | Field component indices (SSOT), field-type mapping, operations, bubble/EM field state |
| `kwavers-signal` | 3.0.0 | Excitation waveforms, pulses, frequency sweeps, modulation, windowing, filters |
| `kwavers-medium` | 4.0.0 | Homogeneous/heterogeneous media; acoustic, elastic, optical, thermal, viscous properties |
| `kwavers-mesh` | 3.0.0 | Tetrahedral FEM meshes: nodes, connectivity, quality metrics, gaia bridge |
| `kwavers-phantom` | 3.0.0 | Tissue-phantom builders: blood oxygenation, layered tissue, tumour, vascular presets |
| `kwavers-boundary` | 3.0.0 | CPML/PML absorbing layers, FEM/BEM, periodic boundaries, smoothing |
| `kwavers-source` | 3.0.0 | Excitation primitives: source trait, grid/mask sources, wavefronts, apodization |
| `kwavers-receiver` | 3.0.0 | Recording primitives: sensor-array geometry, field recorders, point sensors, grid sampling |
| `kwavers-transducer` | 4.1.0 | Devices: focused bowls, phased/linear/matrix/2-D/hemispherical arrays, PAM, ultrafast |
| `kwavers-imaging` | 3.0.0 | DICOM/CT/NIfTI loaders, ultrasound/photoacoustic modalities, CEUS orchestration, fusion |
| `kwavers-physics` | 3.0.0 | Nonlinear acoustics, bubble dynamics, thermal, optics, chemistry, elastic waves |
| `kwavers-solver` | 3.0.0 | FDTD / PSTD / k-space / Helmholtz, BEM, FWI / RTM / CBS, PINN |
| `kwavers-analysis` | 3.0.0 | Signal processing, beamforming, validation, ML/uncertainty, plotting |
| `kwavers-gpu` | 5.0.0 | Hephaestus-backed provider-generic GPU compute backend; concrete `ComputeBackend` impls |
| `kwavers-simulation` | 3.0.0 | Builders, runners, multi-physics coupling, modality pipelines, backends |
| `kwavers-diagnostics` | 3.0.0 | Reconstruction, multi-modal fusion, Doppler, spectroscopy, functional US, decision support |
| `kwavers-therapy` | 3.0.0 | HIFU / histotripsy / lithotripsy planning, theranostic guidance, dose, safety, regulatory |
| `kwavers-driver` | 0.3.13 | Physics-guided, manufacturing-aware driver-electronics design (leaf above `kwavers-transducer`) |
| `kwavers` | 3.0.0 | Thin top-level app/integration crate: binary + cross-cutting tests/examples/benches (no re-exports) |
| `kwavers-python` | 0.1.0 | PyO3 bindings (`pykwavers`); depends on the layer crates directly; no domain logic; `publish = false` |

`xtask/` is the twenty-fifth workspace member: a build-tool package, also
`publish = false`. Shared registry metadata (edition, authors, license,
repository, homepage, keywords, categories) lives once in the root
[`[workspace.package]`](Cargo.toml) table; members inherit it.

Tyche owns reproducible counter streams, Latin-hypercube and Sobol designs,
online moments, correlation screening, and finite-sample conformal
calibration. Kwavers owns physical-domain transforms, model execution, Leto
array presentation, and domain-specific score definitions. Geometry maps
Tyche designs directly into validated rectangles, disks, and balls through one
single-allocation collector; Analysis and PINN fixed-design code carry no
independent provider algorithms, while model-residual adaptive refinement
remains solver-owned. See
[ADR 043](docs/ADR/043-tyche-uncertainty-provider.md).

Each crate carries its own version and moves independently; see the table above
and [`CHANGELOG.md`](CHANGELOG.md) for release history.

Validation status: `pykwavers` reaches 1-to-1 PSTD parity with k-Wave /
k-wave-python / KWave.jl on the homogeneous-water IVP benchmark (Pearson
r ≥ 0.9999 across 1-D/2-D/3-D; see [Reference Benchmark Coverage](#reference-benchmark-coverage)).

### Python Releases

GitHub Releases tagged `kwavers-python-v<version>` build one locked
Python-3.8-compatible stable-ABI wheel per operating system for Linux, Windows,
and macOS. The workflow installs and imports each wheel as `pykwavers`, verifies
the `kwavers-python` distribution identity and Cargo-owned version, attests and
attaches the exact artifacts, then publishes that same wheel set to PyPI
through OIDC Trusted Publishing.

### Rust Crate Releases

The 23 publishable Rust packages — every workspace member except
`kwavers-python` and `xtask`, both of which set `publish = false` — publish to
crates.io in local dependency order.
The `Crates.io Release` workflow validates a named workspace package on manual
dispatch. After its required first release is bootstrapped and its crates.io
Trusted Publisher is registered, a GitHub Release tagged
`crate-<package>-v<version>` packages, verifies, and publishes the matching
Cargo version with a short-lived OIDC token. Validation runs in a separate
read-only job. The publish job is bound to the GitHub `crates-io` environment.
The PyO3 and `xtask` packages are explicitly non-publishable; all Rust library
releases use their package names.

Detailed history lives in [`CHANGELOG.md`](CHANGELOG.md); current work and gaps
are tracked in [`backlog.md`](backlog.md), [`CHECKLIST.md`](CHECKLIST.md), and
[`gap_audit.md`](gap_audit.md).

### Architecture Overview

Kwavers follows a layered architecture designed for scientific computing:

```
Therapy / Diagnostics → Clinical planning, theranostic guidance, dose & safety
Simulation Layer      → Multi-physics orchestration, modality pipelines
Analysis Layer        → Signal processing, beamforming, imaging algorithms
Solver Layer          → Numerical methods (FDTD, PSTD, k-space, FWI, PINN)
Physics Layer         → Wave equations, bubble dynamics, constitutive relations
Domain Layer          → Problem geometry, materials, sources, sensors
Math Layer            → Linear algebra, FFT, numerical primitives
Core Layer            → Fundamental types, error handling
```

Each layer is its own crate, and the crate dependency graph is an acyclic,
strictly unidirectional DAG: `core → math → {grid, field, signal, medium, mesh,
phantom, boundary, source, receiver, transducer, imaging} → physics → solver →
analysis → gpu → simulation → diagnostics/therapy`. No crate depends on a layer
above it, and `kwavers-core` has no first-party dependencies at all. Layers are
enforced by the manifests, not by convention — `cargo tree` is the check.
Dependencies within the domain band (for example `transducer → receiver`) and
skips down the chain are permitted; upward edges are not.

Cross-repository foundations remain below that DAG. Public
[Asclepius](https://github.com/ryancinsight/asclepius) owns CEM43, Arrhenius
thermal damage, and independent-insult composition. Kwavers converts stored
temperatures to Aequitas quantities at its boundaries and retains spatial
fields, tissue presets, clinical thresholds, and an independent validation
oracle. See [ADR 044](docs/ADR/044-asclepius-response-ownership.md).

Public [Hyperion](https://github.com/ryancinsight/hyperion) owns photon and
optical interaction coefficients, reduced scattering, optical depth,
Beer-Lambert transmission, and the diffusion-derived coefficient laws.
`kwavers-medium` retains tissue identity, refractive index, presets, and maps;
`kwavers-physics` and `kwavers-solver` retain spatial transport algorithms and
photoacoustic coupling. This boundary removes the former `kwavers-optics`
formula module plus the parallel `DiffusionOpticalProperties` and
`OpticalAbsorption` models instead of adding a facade around them. See
[ADR 046](docs/ADR/046-hyperion-optical-transport-ownership.md).

Key architectural decisions:
- **Layer Separation**: Unidirectional dependencies prevent circular imports
- **Domain Purity**: Core entities remain free of application logic
- **Trait-Based Design**: Physics specifications defined as traits for testability
- **Feature Flags**: Optional components (GPU, PINN, API) can be enabled as needed

## 🚀 Quick Start

### Installation

There is no facade crate: `kwavers` re-exports nothing, so depend on the layer
crates you actually use.

```toml
[dependencies]
kwavers-core = "3.0.0"
kwavers-grid = "3.0.0"
kwavers-medium = "4.0.0"
```

Add solver, GPU, or PINN capability by pulling in the crate that owns it:

```toml
[dependencies]
kwavers-solver = { version = "3.0.0", features = ["pinn"] }
kwavers-gpu = { version = "5.0.0", features = ["gpu"] }
```

The three snippets below are the doctests of the `kwavers` crate
([`crates/kwavers/README.md`](crates/kwavers/README.md), included as its crate
documentation), so they are compiled and run by
`cargo test -p kwavers --doc` and cannot drift from the API.

### Example 1: Basic Grid Setup

```rust
use kwavers_grid::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 100³ points at 1 mm isotropic spacing.
    let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3)?;

    assert_eq!((grid.nx, grid.ny, grid.nz), (100, 100, 100));
    assert_eq!(grid.dx, 1e-3);
    Ok(())
}
```

### Example 2: Material Properties

`HomogeneousMedium` carries acoustic, optical, thermal, and cavitation
properties at once. The `water`, `tissue`, `blood`, and `air` constructors fill
them from the reference constants in `kwavers-core`. The general constructor is
`HomogeneousMedium::new(density, sound_speed, mu_a, mu_s_prime, grid)` — the
third and fourth arguments are the *optical* absorption and reduced-scattering
coefficients, not acoustic absorption and B/A.

```rust
use kwavers_core::constants::fundamental::{DENSITY_WATER, SOUND_SPEED_WATER};
use kwavers_grid::Grid;
use kwavers_medium::{CoreMedium, HomogeneousMedium};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3)?;
    let water = HomogeneousMedium::water(&grid);

    // Point accessors come from the `CoreMedium` trait.
    let density = water.density(0, 0, 0); // kg/m³
    let sound_speed = water.sound_speed(0, 0, 0); // m/s
    let impedance = density * sound_speed; // Pa·s/m

    assert_eq!(density, DENSITY_WATER);
    assert_eq!(sound_speed, SOUND_SPEED_WATER);
    assert_eq!(impedance, DENSITY_WATER * SOUND_SPEED_WATER);
    Ok(())
}
```

### Example 3: Basic Acoustic Calculations

```rust
fn main() {
    // Acoustic impedance Z = ρc.
    let impedance_water = 1000.0 * 1500.0_f64; // Pa·s/m
    let impedance_air = 1.2 * 343.0_f64;

    // Pressure reflection coefficient R = (Z₂ − Z₁) / (Z₂ + Z₁).
    let reflection = (impedance_air - impedance_water) / (impedance_air + impedance_water);

    // The water/air interface is very nearly a perfect reflector.
    assert!(reflection < -0.999);
}
```

## 📚 Documentation

### 📖 Documentation

- **[Published Kwavers book](https://ryancinsight.github.io/kwavers/)** - Hosted mdBook site
- **[API Reference](https://docs.rs/kwavers)** - Generated Rust documentation
- **[Examples](crates/kwavers/examples/)** - Runnable end-to-end programs
- **[The Kwavers book](docs/book/)** - Chapters, figures, validation narratives
- **[Architecture Decision Records](docs/ADR/)** - Design decisions (incl. ADR-011 crate split)
- **[Documentation index](docs/README.md)** - What lives under `docs/`

### 🎯 Basic Usage

Runnable programs live in
[`crates/kwavers/examples/`](crates/kwavers/examples/):

```bash
# List available examples
cargo run -p kwavers --example

# Run one
cargo run -p kwavers --example basic_simulation
```

**Basic Test**: Check compilation
```bash
cargo check --workspace
```

### 🏗️ Architecture

Kwavers is structured with layered separation intended to support scientific computing workflows:

```
Clinical Applications    → Research use cases, safety monitoring
Analysis & Imaging       → Signal processing, reconstruction algorithms
Simulation Orchestration → Multi-physics coupling, time integration
Numerical Solvers        → FDTD, PSTD, PINN, spectral methods
Physics Specifications   → Wave equations, constitutive relations
Problem Domain           → Geometry, materials, boundary conditions
Mathematical Primitives  → Linear algebra, FFT, interpolation
Core Infrastructure      → Error handling, memory management
```

The architecture separates concerns while maintaining flexibility for different
research applications. Layer boundaries are enforced by the crate graph: each
band is a crate, and Cargo rejects the upward or circular edge that would break
the ordering above.


## 🤝 Contributing

This is an active research project under development. Contributions are welcome! The codebase follows strict quality standards and clean architecture principles.

### 📝 Development Philosophy

- **Deep Vertical Hierarchy**: Modules organized by domain with clear separation of concerns
- **Single Source of Truth**: Shared accessors, no duplication
- **Architectural Purity**: Unidirectional dependencies, no circular imports

### 🚀 Getting Started

1. **Check Status**: Review [`CHECKLIST.md`](CHECKLIST.md) for current task status
2. **Review Plans**: See [`backlog.md`](backlog.md) for planned work and [`gap_audit.md`](gap_audit.md) for known gaps
3. **Build a crate**: `cargo check -p kwavers-core` (per-crate checks are fast post-split)
4. **Run Tests**: `cargo nextest run -p <crate>`; use `cargo test -p <crate> --doc`
   for doctests
5. **Read Docs**: [`docs/book/`](docs/book/) for narratives, [`docs/ADR/`](docs/ADR/) for design decisions

### 📊 Development Approach

**Artifact-driven sprints**, tracked in the repository-root artifacts:
- [`backlog.md`](backlog.md) — strategy and prioritized work
- [`CHECKLIST.md`](CHECKLIST.md) — tactical tasks with change-class tags
- [`gap_audit.md`](gap_audit.md) — physics/numerics gap findings
- [`CHANGELOG.md`](CHANGELOG.md) — version history

**Quality Standards**:
- Zero compilation errors (enforced)
- 100% test pass rate for all refactoring
- Mathematical specifications with literature references

### 🔬 Research Integration

Kwavers is being enhanced with methods from leading ultrasound simulation projects:
- **k-Wave** (MATLAB): k-space pseudospectral methods, advanced source modeling
- **jwave** (JAX/Python): Differentiable simulations, GPU parallelization
- **k-wave-python**: Python binding patterns, HDF5 standards
- **KWave.jl** (Julia): MATLAB-free k-Wave implementation for 1-D/2-D/3-D reference benchmarking
- **optimus**: Optimization frameworks, inverse problems
- **fullwave25**: Full-wave simulation, clinical workflows
- **dbua**: Neural beamforming, real-time inference
- **simsonic**: Advanced tissue models, multi-modal integration

### 📊 History

Per-version history is consolidated in [`CHANGELOG.md`](CHANGELOG.md). The
historical per-sprint/per-phase reports formerly under `docs/` were pruned
during the workspace-split docs cleanup and remain recoverable from git history.

## 📄 License

Kwavers is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📚 References

### Key Physics Texts
- Hamilton, M.F. & Blackstock, D.T. - Nonlinear Acoustics
- Szabo, T.L. - Diagnostic Ultrasound Imaging
- Duck, F.A. - Physical Properties of Tissues

### Numerical Methods
- Yee, K.S. (1966) - FDTD method
- Liu, Q.H. (1997) - PSTD method
- Hesthaven, J.S. (2007) - DG methods

### Related Simulation Projects
- **k-Wave**: MATLAB toolbox for acoustic wave simulation ([GitHub](https://github.com/ucl-bug/k-wave))
- **jwave**: JAX-based differentiable acoustic simulations ([GitHub](https://github.com/ucl-bug/jwave))
- **k-wave-python**: Python interface to k-Wave ([GitHub](https://github.com/waltsims/k-wave-python))
- **KWave.jl**: Julia implementation of k-Wave for MATLAB-free acoustic simulation benchmarks ([GitHub](https://github.com/JClingo/k-wave-julia))
- **optimus**: Optimization framework for ultrasound ([GitHub](https://github.com/optimuslib/optimus))
- **fullwave25**: Full-wave ultrasound simulator ([GitHub](https://github.com/pinton-lab/fullwave25))
- **dbua**: Deep learning beamforming ([GitHub](https://github.com/waltsims/dbua))
- **simsonic**: Advanced ultrasound simulation platform ([Website](https://www.simsonic.fr))

### Reference Benchmark Coverage

The MATLAB-free benchmark harness in `external/k-wave-julia/benchmarks/kwavers`
compares the same homogeneous-water IVP Gaussian source case across KWave.jl,
k-wave-python, and pykwavers for 1-D, 2-D, and 3-D. Native MATLAB k-Wave source
is present in `external/k-wave`, but it is not executed unless MATLAB or Octave
is available.

| Dimension | KWave.jl | k-wave-python | pykwavers | Current result |
|-----------|----------|---------------|-----------|----------------|
| 1-D | Native `KWaveGrid(nx, dx)` | Native Python backend | `(nx, 1, 1)` active grid | PASS: k-wave-python r=0.999977, pykwavers r=0.999976 |
| 2-D | Native `KWaveGrid(nx, dx, nx, dx)` | Native Python backend | `(nx, nx, 1)` active grid | PASS: k-wave-python r=0.999948, pykwavers r=0.999948 |
| 3-D | Native `KWaveGrid(nx, dx, nx, dx, nx, dx)` | Native Python backend | `(nx, nx, nx)` active grid | PASS: k-wave-python r=0.999909, pykwavers r=0.999909 |
| MATLAB k-Wave | Source available | Not applicable | Not applicable | Not run without MATLAB/Octave |

### Key Publications
1. Treeby & Cox (2010) - "k-Wave: MATLAB toolbox for photoacoustic simulation" - J. Biomed. Opt. 15(2), 021314
2. Treeby et al. (2012) - "Nonlinear ultrasound propagation in heterogeneous media" - J. Acoust. Soc. Am. 131(6), 4324-4336
3. Wise et al. (2019) - "Arbitrary acoustic source distributions" - J. Acoust. Soc. Am. 146(1), 278-288
4. Treeby et al. (2020) - "Axisymmetric k-space method" - J. Acoust. Soc. Am. 148(4), 2288-2300

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/ryancinsight/kwavers/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ryancinsight/kwavers/discussions)
- **Documentation**: [docs.rs/kwavers](https://docs.rs/kwavers)

---

**A research library for acoustic and optical physics simulations.**
