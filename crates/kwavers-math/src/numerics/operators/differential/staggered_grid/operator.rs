//! `StaggeredGridOperator` struct and constructor.
//!
//! SRP: changes when the operator contract or the trait surface changes.

use kwavers_core::error::{KwaversResult, NumericalError};
use leto::{Array3, ArrayView3};

/// Staggered grid finite difference operator (Yee scheme).
///
/// Field components are offset by half a grid cell, providing natural
/// conservation properties and second-order accuracy.
#[derive(Debug)]
pub struct StaggeredGridOperator {
    /// Grid spacing along x \[m\]. `pub` for cross-crate consumers (e.g. the FDTD
    /// velocity updater in `kwavers-solver`) that read the operator's spacing.
    pub dx: f64,
    /// Grid spacing along y \[m\].
    pub dy: f64,
    /// Grid spacing along z \[m\].
    pub dz: f64,
}

impl StaggeredGridOperator {
    /// Create a new staggered grid operator.
    ///
    /// # Errors
    ///
    /// Returns `InvalidGridSpacing` if any spacing is non-positive.
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }
}

impl StaggeredGridOperator {
    /// Order of the staggered scheme (2nd-order accurate).
    pub fn order(&self) -> usize {
        2
    }

    /// Stencil width of the staggered scheme (2 points).
    pub fn stencil_width(&self) -> usize {
        2
    }

    /// Whether the scheme is conservative.
    pub fn is_conservative(&self) -> bool {
        true
    }
}
