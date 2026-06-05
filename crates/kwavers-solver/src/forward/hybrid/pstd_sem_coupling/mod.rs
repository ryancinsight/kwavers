//! PSTD-SEM Coupling Implementation
//!
//! Implements coupling between Pseudo-Spectral Time Domain (PSTD)
//! and Spectral Element Method (SEM) solvers for high-accuracy acoustic simulations.
//!
//! ## Mathematical Foundation
//!
//! Both PSTD and SEM are spectral methods with exponential convergence:
//!
//! ```text
//! PSTD: Global spectral accuracy via FFT
//!       u(x,t+Δt) = F^{-1}[e^{ik·Δx·c·Δt} · F[u(x,t)]]
//!
//! SEM: Local spectral accuracy via nodal basis
//!      uₕ(x,t) = ∑ᵢ uᵢ(t) · φᵢ(x)  within each element
//! ```
//!
//! ## Literature References
//!
//! - Kopriva, D. A. (2009). "Implementing spectral methods for partial differential equations"
//! - Hesthaven, J. S., & Warburton, T. (2008). "Nodal discontinuous Galerkin methods"
//! - Liu, Q. H. (1997). "The PSTD algorithm: A time-domain method requiring only two cells per wavelength"

mod coupler;
mod interface;
mod solver;
#[cfg(test)]
mod tests;

pub use coupler::PstdSemCoupler;
pub use solver::PstdSemSolver;

use ndarray::Array2;

/// Configuration for PSTD-SEM coupling
#[derive(Debug, Clone)]
pub struct PstdSemCouplingConfig {
    /// Overlap region thickness (elements/cells)
    pub overlap_thickness: usize,
    /// Modal coupling order (polynomial degree for interface)
    pub coupling_order: usize,
    /// Conservative projection tolerance
    pub projection_tolerance: f64,
    /// Interface stabilization parameter
    pub stabilization_alpha: f64,
}

impl Default for PstdSemCouplingConfig {
    fn default() -> Self {
        Self {
            overlap_thickness: 2,
            coupling_order: 4,
            projection_tolerance: 1e-12,
            stabilization_alpha: 0.1,
        }
    }
}

/// Spectral coupling interface between PSTD and SEM domains
#[derive(Debug)]
pub struct SpectralCouplingInterface {
    /// PSTD grid points at interface
    pub(super) pstd_interface_points: Vec<(usize, usize, usize)>,
    /// SEM nodes at interface
    pub(super) sem_interface_nodes: Vec<usize>,
    /// Modal transformation matrix (PSTD spectral → SEM modal)
    pub(super) modal_transform: Array2<f64>,
    /// Conservative projection matrix (SEM → PSTD)
    pub(super) projection_matrix: Array2<f64>,
}
