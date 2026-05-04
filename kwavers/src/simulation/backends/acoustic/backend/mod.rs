//! Backend abstraction trait for acoustic wave solvers
//!
//! This module defines the trait interface for acoustic solver backends,
//! enabling the simulation layer to delegate to different numerical methods
//! (FDTD, PSTD, etc.) while maintaining a unified API.
//!
//! # Design Pattern
//!
//! The `AcousticSolverBackend` trait follows the **Strategy Pattern**, allowing
//! runtime selection of the appropriate numerical solver based on problem
//! characteristics:
//!
//! - **FDTD** (Finite-Difference Time-Domain): Robust for heterogeneous media,
//!   handles discontinuities well, straightforward implementation
//! - **PSTD** (Pseudospectral Time-Domain): Spectral accuracy for smooth media,
//!   4-8x fewer grid points, efficient for homogeneous cases
//! - **Nonlinear** (Westervelt/KZK): High-intensity therapeutic ultrasound,
//!   shock formation, harmonic generation
//!
//! # Backend Selection Criteria
//!
//! | Criterion | FDTD | PSTD |
//! |-----------|------|------|
//! | Heterogeneity | High (>30%) | Low (<30%) |
//! | Points per wavelength | 4-10 | 2-4 |
//! | Discontinuities | Sharp interfaces | Smooth variations |
//! | Efficiency | O(N log N) | O(N log N) |
//!
//! # Trait Design Principles
//!
//! - **Minimal Interface**: Only essential operations for therapy applications
//! - **Zero-Cost Abstraction**: Trait object overhead acceptable for long-running simulations
//! - **Thread Safety**: Not required (single-threaded simulation loop)
//! - **Error Handling**: All fallible operations return `KwaversResult`
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use crate::simulation::backends::AcousticSolverBackend;
//!
//! fn run_simulation(mut backend: Box<dyn AcousticSolverBackend>) -> KwaversResult<()> {
//!     // Time stepping loop
//!     for _ in 0..1000 {
//!         backend.step()?;
//!
//!         // Monitor pressure field
//!         let p = backend.get_pressure_field();
//!         let p_max = p.iter().cloned().fold(0.0, f64::max);
//!         println!("Max pressure: {:.2} kPa", p_max / 1e3);
//!     }
//!
//!     // Compute intensity for safety validation
//!     let intensity = backend.get_intensity_field()?;
//!     Ok(())
//! }
//! ```

#[cfg(test)]
mod tests;

use crate::core::error::KwaversResult;
use crate::domain::mesh::BoundaryType;
use crate::domain::source::Source;
use ndarray::Array3;
use ndarray::{Array1, ArrayView2};
use num_complex::Complex64;
use std::fmt::Debug;
use std::sync::Arc;

/// Backend trait for acoustic wave solvers.
///
/// Defines the interface that all acoustic solver backends must implement
/// to be usable by the simulation layer. This trait enables polymorphic solver
/// selection based on problem characteristics.
///
/// # Mathematical Foundations
///
/// All backends must solve the acoustic wave equations in either first-order
/// (pressure-velocity) or second-order (pressure only) form:
///
/// **First-Order System** (FDTD):
/// ```text
/// ∂v/∂t = -(1/ρ₀)∇p        (momentum conservation)
/// ∂p/∂t = -ρ₀c₀²∇·v        (mass conservation)
/// ```
///
/// **Second-Order Form** (PSTD):
/// ```text
/// ∇²p - (1/c₀²)∂²p/∂t² = 0
/// ```
///
/// # Stability Requirements
///
/// - **FDTD CFL**: `c_max·Δt/Δx ≤ 1/√3` (3D)
/// - **PSTD**: `c_max·Δt·k_max ≤ π` where `k_max = π/Δx`
pub trait AcousticSolverBackend: Debug {
    /// Advance simulation by one time step.
    ///
    /// Updates the pressure and velocity fields by integrating the acoustic
    /// wave equations forward by the backend's internal time step `dt`.
    fn step(&mut self) -> KwaversResult<()>;

    /// Get current pressure field (Pa).
    ///
    /// Array dimensions are `[nx, ny, nz]`. Physical position of `field[[i,j,k]]`
    /// is `(i·dx, j·dy, k·dz)`.
    fn get_pressure_field(&self) -> &Array3<f64>;

    /// Get current particle velocity fields (m/s).
    ///
    /// Returns references to `(vx, vy, vz)`. Lifetimes tied to backend.
    fn get_velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>);

    /// Get acoustic intensity field (W/m²).
    ///
    /// Plane wave approximation: `I = p² / (ρ₀c₀)`.
    fn get_intensity_field(&self) -> KwaversResult<Array3<f64>>;

    /// Get acoustic impedance field `Z = ρ₀c₀` (kg/(m²·s)).
    fn get_impedance_field(&self) -> KwaversResult<Array3<f64>>;

    /// Get simulation time step (s).
    ///
    /// Determined during initialization to satisfy the CFL stability condition.
    fn get_dt(&self) -> f64;

    /// Add dynamic source to simulation.
    ///
    /// Source is evaluated and applied at each subsequent time step.
    fn add_source(&mut self, source: Arc<dyn Source>) -> KwaversResult<()>;

    /// Get current simulation time (s): `t = (steps_completed) · dt`.
    fn get_current_time(&self) -> f64;

    /// Get grid dimensions `(nx, ny, nz)`.
    fn get_grid_dimensions(&self) -> (usize, usize, usize);
}

/// Backend trait for frequency-domain acoustic Helmholtz solves.
///
/// This trait is intentionally separate from [`AcousticSolverBackend`]. A
/// Helmholtz solve computes complex steady-state pressure for one angular
/// frequency and has no timestep, velocity field, or CFL progression.
pub trait FrequencyDomainAcousticBackend: Debug {
    /// Queue an exact nodal load `F_i += value`.
    fn add_nodal_load(&mut self, node_idx: usize, value: Complex64) -> KwaversResult<()>;

    /// Queue Dirichlet values on nodes tagged with a mesh boundary type.
    fn add_dirichlet_on_boundary_type(
        &mut self,
        boundary_type: BoundaryType,
        value: Complex64,
    ) -> KwaversResult<usize>;

    /// Assemble and solve the frequency-domain system.
    fn solve(&mut self) -> KwaversResult<()>;

    /// Borrow the complex nodal pressure solution.
    fn pressure_solution(&self) -> &Array1<Complex64>;

    /// Interpolate complex pressure at physical query points.
    fn interpolate_pressure(
        &self,
        query_points: ArrayView2<'_, f64>,
    ) -> KwaversResult<Array1<Complex64>>;

    /// Wavenumber used by the Helmholtz operator.
    fn wavenumber(&self) -> f64;

    /// Mesh size as `(nodes, tetrahedra)`.
    fn mesh_size(&self) -> (usize, usize);
}
