//! Diffusion Approximation Solver for Optical Fluence Computation
//!
//! # Mathematical Foundation
//!
//! ## Steady-State Diffusion Equation
//!
//! For continuous-wave (CW) illumination, the photon fluence Φ(r) satisfies:
//!
//! ```text
//! ∇·(D(r)∇Φ(r)) - μₐ(r)Φ(r) = -S(r)
//! ```
//!
//! Where:
//! - `Φ(r)`: Optical fluence (W/m²)
//! - `D(r) = 1/(3(μₐ + μₛ'))`: Diffusion coefficient (m)
//! - `μₐ(r)`: Absorption coefficient (m⁻¹)
//! - `μₛ'(r) = μₛ(1-g)`: Reduced scattering coefficient (m⁻¹)
//! - `S(r)`: Isotropic source term (W/m³)
//!
//! ## Boundary Conditions (Extrapolated Boundary)
//!
//! At tissue-air interface, partial current boundary condition:
//!
//! ```text
//! Φ(r_b) + 2A D(r_b) ∂Φ/∂n|_{r_b} = 0
//! ```
//!
//! Where `A = (1 + R_eff)/(1 - R_eff)` accounts for internal reflection.
//! For typical tissue-air interface (n=1.4), `A ≈ 2.0`.
//!
//! ## Discretization (Finite Difference Method)
//!
//! Second-order central differences on uniform Cartesian grid:
//!
//! ```text
//! ∇·(D∇Φ) ≈ (D_{i+1/2}(Φ_{i+1} - Φᵢ) - D_{i-1/2}(Φᵢ - Φ_{i-1}))/Δx²
//! ```
//!
//! Results in 7-point stencil for 3D (19-point for heterogeneous D).
//!
//! # Module layout
//!
//! - [`construction`]: `new` / `uniform` constructors and the
//!   `boundary_conditions`/`ghost_coefficient` helpers.
//! - [`operator`]: `apply_operator` — `A Φ = ∇·(D∇Φ) − μₐΦ` 7-point
//!   stencil with extrapolated-boundary handling per face.
//! - [`preconditioner`]: Jacobi preconditioner (inverse diagonal of `A`).
//! - [`solve`]: preconditioned conjugate-gradient driver loop.
//! - [`accessors`]: `grid` / `diffusion_coefficient` / `absorption_coefficient`
//!   read-only views.
//! - [`analytical`]: Green's-function reference solutions for tests
//!   (infinite + semi-infinite medium).
//!
//! ## References
//!
//! - **Arridge (1999)**: "Optical tomography in medical imaging." *Inverse Problems*
//! - **Wang & Jacques (1995)**: "Monte Carlo modeling of light transport." *Computer Methods*
//! - **Contini et al. (1997)**: "Photon migration through a turbid slab." *Applied Optics*

mod accessors;
pub mod analytical;
mod construction;
mod operator;
mod preconditioner;
mod solve;

#[cfg(test)]
mod tests;

use ndarray::Array3;

use crate::domain::grid::Grid;

/// Configuration for diffusion solver
#[derive(Debug, Clone)]
pub struct DiffusionSolverConfig {
    /// Maximum number of conjugate gradient iterations
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual)
    pub tolerance: f64,
    /// Extrapolated boundary parameter A (default 2.0 for tissue-air)
    pub boundary_parameter: f64,
    pub boundary_conditions: Option<DiffusionBoundaryConditions>,
    /// Enable verbose convergence logging
    pub verbose: bool,
}

impl Default for DiffusionSolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            tolerance: 1e-6,
            boundary_parameter: 2.0,
            boundary_conditions: None,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DiffusionBoundaryCondition {
    ZeroFlux,
    Extrapolated { a: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct DiffusionBoundaryConditions {
    pub x_min: DiffusionBoundaryCondition,
    pub x_max: DiffusionBoundaryCondition,
    pub y_min: DiffusionBoundaryCondition,
    pub y_max: DiffusionBoundaryCondition,
    pub z_min: DiffusionBoundaryCondition,
    pub z_max: DiffusionBoundaryCondition,
}

impl DiffusionBoundaryConditions {
    #[must_use]
    pub fn all_extrapolated(a: f64) -> Self {
        Self {
            x_min: DiffusionBoundaryCondition::Extrapolated { a },
            x_max: DiffusionBoundaryCondition::Extrapolated { a },
            y_min: DiffusionBoundaryCondition::Extrapolated { a },
            y_max: DiffusionBoundaryCondition::Extrapolated { a },
            z_min: DiffusionBoundaryCondition::Extrapolated { a },
            z_max: DiffusionBoundaryCondition::Extrapolated { a },
        }
    }
}

impl Default for DiffusionBoundaryConditions {
    fn default() -> Self {
        Self::all_extrapolated(2.0)
    }
}

/// Steady-state diffusion solver for optical fluence
///
/// Solves: `∇·(D(r)∇Φ(r)) - μₐ(r)Φ(r) = -S(r)`
/// using finite difference discretization and conjugate gradient iteration.
#[derive(Debug)]
pub struct DiffusionSolver {
    /// Computational grid
    pub(super) grid: Grid,
    /// Spatial diffusion coefficient field `D(r) = 1/(3(μₐ + μₛ'))`
    pub(super) diffusion_coefficient: Array3<f64>,
    /// Absorption coefficient field `μₐ(r)`
    pub(super) absorption_coefficient: Array3<f64>,
    /// Solver configuration
    pub(super) config: DiffusionSolverConfig,
}
