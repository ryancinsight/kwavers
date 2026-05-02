//! Schwarz domain decomposition boundary condition
//!
//! Implements transmission conditions for domain decomposition methods,
//! enabling parallel solution of large problems via overlapping subdomains.
//!
//! # Mathematical Foundation
//!
//! In domain decomposition, a large computational domain Ω is divided into
//! overlapping subdomains Ω₁, Ω₂, ..., Ωₙ with overlap regions. The Schwarz
//! boundary conditions enforce consistency at subdomain interfaces.
//!
//! ## Transmission Conditions
//!
//! Four canonical transmission conditions are dispatched in `transmission.rs`:
//!
//! - **Dirichlet**: `u₁ = u₂` on the interface (direct value copy).
//! - **Neumann**: `∂u₁/∂n = ∂u₂/∂n` (flux continuity via gradient correction).
//! - **Robin**: `∂u/∂n + αu = β` (impedance / convective boundary).
//! - **Optimized**: relaxation-weighted update `u_new = (1−θ)u_old + θ u_neighbor`.
//!
//! # Module layout
//!
//! - [`gradient`]: shared `compute_normal_gradient` finite-difference helper.
//! - [`transmission`]: branching `apply_transmission` dispatcher.
//! - [`boundary_impl`]: `BoundaryCondition` trait bridge for the framework.
//!
//! # References
//!
//! - Schwarz, H. A. (1870). *Über einen Grenzübergang durch alternirendes Verfahren*.
//! - Lions, P.-L. (1988). "On the Schwarz alternating method I." *Domain decomposition*.
//! - Gander, M. J. (2006). "Optimized Schwarz methods." *SIAM J. Numer. Anal.*
//! - Quarteroni, A. & Valli, A. (1999). "Domain Decomposition Methods for PDEs"

mod boundary_impl;
mod gradient;
mod transmission;

#[cfg(test)]
mod tests;

use super::types::{BoundaryDirections, TransmissionCondition};

/// Schwarz domain decomposition boundary
///
/// Implements transmission conditions for domain decomposition methods,
/// enabling parallel solution of large problems. Named after Hermann Amandus
/// Schwarz, who introduced the alternating method in 1870.
#[derive(Debug, Clone)]
pub struct SchwarzBoundary {
    /// Overlap region thickness in meters
    pub overlap_thickness: f64,
    /// Transmission condition type
    pub transmission_condition: TransmissionCondition,
    /// Relaxation parameter θ for optimized Schwarz (0 < θ ≤ 1)
    pub relaxation_parameter: f64,
    /// Boundary directions
    pub directions: BoundaryDirections,
}

impl SchwarzBoundary {
    /// Create a new Schwarz boundary
    ///
    /// # Arguments
    ///
    /// * `overlap_thickness` - Thickness of overlap region in meters
    /// * `directions` - Boundary directions to apply
    ///
    /// # Returns
    ///
    /// New `SchwarzBoundary` with Dirichlet transmission (default)
    pub fn new(overlap_thickness: f64, directions: BoundaryDirections) -> Self {
        Self {
            overlap_thickness,
            transmission_condition: TransmissionCondition::Dirichlet,
            relaxation_parameter: 1.0,
            directions,
        }
    }

    /// Set transmission condition.
    ///
    /// # Arguments
    ///
    /// * `condition` - Transmission condition type (Dirichlet, Neumann, Robin, Optimized)
    pub fn with_transmission_condition(mut self, condition: TransmissionCondition) -> Self {
        self.transmission_condition = condition;
        self
    }

    /// Set relaxation parameter for optimized Schwarz.
    ///
    /// # Arguments
    ///
    /// * `relaxation` - Relaxation parameter θ ∈ (0, 1]
    ///   - θ = 1: Full update (no relaxation)
    ///   - θ < 1: Under-relaxation (slower but more stable)
    pub fn with_relaxation(mut self, relaxation: f64) -> Self {
        self.relaxation_parameter = relaxation;
        self
    }
}
