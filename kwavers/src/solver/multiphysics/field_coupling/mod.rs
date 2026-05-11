//! Field coupling strategies for multi-physics simulations
//!
//! This module implements different strategies for coupling fields between
//! different physics domains (acoustic, optical, thermal).

mod coupling_ops;
#[cfg(test)]
mod tests;

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Field coupling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CouplingStrategy {
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
pub struct FieldCoupler {
    /// Coupling strategy
    strategy: CouplingStrategy,
    /// Coupling strength parameters
    coupling_strength: f64,
    /// Maximum iterations for strong coupling
    max_iterations: usize,
    /// Tolerance for convergence
    tolerance: f64,
}

impl FieldCoupler {
    /// Create a new field coupler
    #[must_use]
    pub fn new(strategy: CouplingStrategy) -> Self {
        Self {
            strategy,
            coupling_strength: 1.0,
            max_iterations: 10,
            tolerance: 1e-6,
        }
    }

    /// Couple fields according to the selected strategy
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn couple_fields(&self, fields: &mut [Array3<f64>], dt: f64) -> KwaversResult<()> {
        match self.strategy {
            CouplingStrategy::None => Ok(()),
            CouplingStrategy::Weak => self.apply_weak_coupling(fields, dt),
            CouplingStrategy::Strong => self.apply_strong_coupling(fields, dt),
            CouplingStrategy::Adaptive => self.apply_adaptive_coupling(fields, dt),
        }
    }
}
