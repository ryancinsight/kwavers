//! Periodic Boundary Conditions
//!
//! This module implements periodic boundary conditions for acoustic simulations,
//! essential for standing wave validation and resonance studies.
//!
//! # Mathematical Specification
//!
//! Periodic boundaries enforce:
//!
//! ```text
//! p(x + L, y, z, t) = p(x, y, z, t)    (1) Periodic in x
//! p(x, y + L, z, t) = p(x, y, z, t)    (2) Periodic in y
//! p(x, y, z + L, t) = p(x, y, z, t)    (3) Periodic in z
//! ```
//!
//! ## Standing Wave Application
//!
//! For a 1D standing wave with periodic boundaries:
//!
//! ```text
//! p(x, t) = 2A sin(kx) cos(ωt)         (4)
//! with resonance condition: k = nπ/L, n ∈ ℕ  (5)
//! ```
//!
//! ## Bloch Periodic Boundaries
//!
//! For metamaterials and phononic crystals, Bloch periodic boundaries with phase shift:
//! `p(x + L) = p(x) exp(ik·L)` (6)
//!
//! # References
//!
//! 1. Pierce, A. D. (1989). *Acoustics*, Ch. 5.
//! 2. Brillouin, L. (1953). *Wave Propagation in Periodic Structures*.
//! 3. Treeby & Cox (2010). "k-Wave: MATLAB toolbox".

use kwavers_core::error::{KwaversError, KwaversResult};
use crate::boundary::traits::BoundaryDirections;

/// Periodic boundary condition configuration
#[derive(Debug, Clone)]
pub struct PeriodicConfig {
    pub periodic_x: bool,
    pub periodic_y: bool,
    pub periodic_z: bool,
    /// Bloch wave vector phase shift k·L (rad) for each direction.
    /// For standard periodic boundaries, use [0.0, 0.0, 0.0].
    pub bloch_phase: [f64; 3],
}

impl Default for PeriodicConfig {
    fn default() -> Self {
        Self {
            periodic_x: true,
            periodic_y: true,
            periodic_z: true,
            bloch_phase: [0.0, 0.0, 0.0],
        }
    }
}

impl PeriodicConfig {
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn new(periodic_x: bool, periodic_y: bool, periodic_z: bool) -> Self {
        Self {
            periodic_x,
            periodic_y,
            periodic_z,
            bloch_phase: [0.0, 0.0, 0.0],
        }
    }
    /// With bloch phase.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_bloch_phase(mut self, bloch_phase: [f64; 3]) -> Self {
        self.bloch_phase = bloch_phase;
        self
    }
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub(super) fn validate(&self) -> KwaversResult<()> {
        for &phase in &self.bloch_phase {
            if !phase.is_finite() {
                return Err(KwaversError::Validation(
                    kwavers_core::error::validation::ValidationError::ConstraintViolation {
                        message: "Bloch phase must be finite".to_owned(),
                    },
                ));
            }
        }
        Ok(())
    }
}

/// Periodic boundary condition implementation
///
/// Resonant frequencies for periodic domain of length L:
/// `f_n = n·c₀/(2L), n = 1, 2, 3, ...` (Pierce 1989, Ch. 5)
#[derive(Debug, Clone)]
pub struct PeriodicBoundaryCondition {
    pub(super) config: PeriodicConfig,
    pub(super) active_directions: BoundaryDirections,
}

#[cfg(test)]
mod tests;
mod wrapping;
