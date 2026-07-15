//! Field coupling strategies for multi-physics simulations
//!
//! This module implements different strategies for coupling fields between
//! different physics domains (acoustic, optical, thermal).

mod coupling_ops;
#[cfg(test)]
mod tests;

use kwavers_core::error::KwaversResult;
use leto::Array3;

/// Field coupling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldCouplingStrategy {
    /// No coupling between fields (independent simulations)
    None,
    /// Weak coupling (sequential updates)
    Weak,
    /// Strong coupling (iterative updates)
    Strong,
    /// Adaptive coupling (adjusts based on field gradients)
    Adaptive,
}

/// Field coupler for multi-physics interactions
#[derive(Debug)]
pub struct MultiphysicsFieldCoupler {
    /// Coupling strategy
    strategy: FieldCouplingStrategy,
    /// Coupling strength parameters
    coupling_strength: f64,
    /// Maximum iterations for strong coupling
    max_iterations: usize,
    /// Tolerance for convergence
    tolerance: f64,
}

impl MultiphysicsFieldCoupler {
    /// Create a new field coupler
    #[must_use]
    pub fn new(strategy: FieldCouplingStrategy) -> Self {
        Self {
            strategy,
            coupling_strength: 1.0,
            max_iterations: 10,
            tolerance: 1e-6,
        }
    }

    /// Couple fields according to the selected strategy
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn couple_fields(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        match self.strategy {
            FieldCouplingStrategy::None => Ok(()),
            FieldCouplingStrategy::Weak => self.apply_weak_coupling(fields, dt),
            FieldCouplingStrategy::Strong => self.apply_strong_coupling(fields, dt),
            FieldCouplingStrategy::Adaptive => self.apply_adaptive_coupling(fields, dt),
        }
    }
}
