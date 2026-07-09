//! Shear Wave Elastography (SWE) Solver
//!
//! Numerical solver for elastic wave propagation in soft tissue for elastography applications.
//!
//! ## Architecture
//!
//! This module correctly separates concerns:
//! - **solver/forward/elastic/swe/**: Numerical methods (THIS MODULE)
//! - **physics/acoustics/imaging/modalities/elastography/**: Physics models and material properties
//!
//! ## Module Organization
//!
//! The SWE solver is organized into focused submodules following GRASP principles:
//!
//! - `types`: Configuration types, enums, and data structures
//! - `stress`: Stress tensor derivative computations
//! - `integration`: Time integration schemes (velocity-Verlet)
//! - `boundary`: PML boundary conditions
//! - `core`: Main solver orchestration (to be implemented)
//!
//! ## Mathematical Foundation
//!
//! ### Elastic Wave Equation
//!
//! The solver implements the elastic wave equation for displacement `u`:
//!
//! ```text
//! ρ ∂²u/∂t² = (λ + μ)∇(∇·u) + μ∇²u + f
//! ```
//!
//! Where:
//! - `ρ`: Density (kg/m³)
//! - `u`: Displacement vector (m)
//! - `λ, μ`: Lamé parameters (Pa)
//! - `f`: Body force (N/m³)
//!
//! ### First-Order Form
//!
//! Converted to first-order system for time integration:
//!
//! ```text
//! ∂u/∂t = v
//! ρ ∂v/∂t = ∇·σ + f
//! ```
//!
//! Where:
//! - `v`: Velocity (m/s)
//! - `σ`: Stress tensor (Pa)
//!
//! ### Stress-Strain Relation (Hooke's Law)
//!
//! ```text
//! σij = λ δij εkk + 2μ εij
//! ```
//!
//! Where:
//! - `εij = ½(∂ui/∂xj + ∂uj/∂xi)`: Strain tensor
//! - `δij`: Kronecker delta
//!
//! ### Lamé Parameters
//!
//! Related to engineering constants:
//!
//! ```text
//! λ = Eν / ((1+ν)(1-2ν))
//! μ = E / (2(1+ν))
//! ```
//!
//! For nearly incompressible tissue (ν ≈ 0.5):
//!
//! ```text
//! E ≈ 3μ
//! cs = sqrt(μ/ρ)  (shear wave speed)
//! ```
//!
//! ## Numerical Method
//!
//! - **Spatial Discretization**: 4th-order finite differences
//! - **Time Integration**: Velocity-Verlet (2nd-order symplectic)
//! - **Boundary Conditions**: Perfectly Matched Layer (PML)
//! - **Stability**: CFL condition `Δt < Δx/(√3·cmax)`
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use kwavers_solver::forward::elastic::swe::{
//!     ElasticWaveSolver, ElasticWaveConfig
//! };
//! use kwavers_grid::Grid;
//! use kwavers_medium::HomogeneousMedium;
//!
//! # fn main() -> kwavers_core::error::KwaversResult<()> {
//! // Create computational grid
//! let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3)?;
//!
//! // Define tissue properties
//! let medium = HomogeneousMedium::new(kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL, kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM, 0.0, 0.0, &grid);
//!
//! // Configure solver
//! let config = ElasticWaveConfig::default();
//!
//! // Create solver (to be implemented)
//! // let solver = ElasticWaveSolver::new(&grid, &medium, config)?;
//!
//! // Run simulation
//! // let fields = solver.propagate(&leto::Array3::zeros((64, 64, 64)))?;
//! # Ok(())
//! # }
//! ```
//!
//! ## References
//!
//! - Moczo, P., et al. (2007). "3D finite-difference method for elastodynamics."
//!   *Solid Earth*, 93(3), 523-553.
//! - Komatitsch, D., & Martin, R. (2007). "Unsplit PML for seismic waves."
//!   *Geophysics*, 72(5), SM155-SM167.
//! - Palmeri, M. L., et al. (2008). "Quantifying hepatic shear modulus in vivo."
//!   *Ultrasound Med Biol*, 34(4), 546-558.

// Submodules (alphabetical order)
pub mod boundary;
pub mod core;
pub mod gpu;
pub mod integration;
pub mod scratch;
pub mod stress;
pub mod types;

// Re-export public API types for ergonomic imports
pub use types::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticVelocitySource, ElasticVelocitySourceMode,
    ElasticWaveConfig, ElasticWaveField, VolumetricQualityMetrics, VolumetricSource,
    VolumetricWaveConfig, WaveFrontTracker,
};

pub use boundary::{ElasticSwePMLBoundary, SwePmlConfig};
pub use core::{ElasticPointForce, ElasticWaveSolver};
pub use gpu::{AdaptiveResolution, GPUDevice, GPUElasticWaveSolver3D};
pub use integration::TimeIntegrator;
pub use scratch::ElasticStepScratch;

// Note: The elastic wave solver was originally misplaced in:
// physics/acoustics/imaging/modalities/elastography/elastic_wave_solver.rs
//
// This has been corrected to follow proper architectural separation:
// - Numerical solvers belong in solver/forward/elastic/
// - Physics models belong in physics/acoustics/imaging/modalities/elastography/
//
// Refactoring Progress:
// ✅ types.rs - Configuration and data types
// ✅ stress.rs - Stress tensor derivatives
// ✅ integration.rs - Time integration schemes
// ✅ boundary.rs - PML boundary conditions
// ✅ core.rs - Main solver orchestration
// ✅ gpu.rs - GPU acceleration support
// 🔄 tracking.rs - Wave-front tracking (planned)
