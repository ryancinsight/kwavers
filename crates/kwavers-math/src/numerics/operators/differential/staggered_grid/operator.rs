//! `StaggeredGridOperator` struct, constructor, and `DifferentialOperator` impl.
//!
//! SRP: changes when the operator contract or the trait surface changes.

use super::super::DifferentialOperator;
use kwavers_core::error::{KwaversResult, NumericalError};
use ndarray::{Array3, ArrayView3};

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

impl DifferentialOperator for StaggeredGridOperator {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.apply_forward_x(field)
    }
    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.apply_forward_y(field)
    }
    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.apply_forward_z(field)
    }
    fn order(&self) -> usize {
        2
    }
    fn stencil_width(&self) -> usize {
        2
    }
    fn is_conservative(&self) -> bool {
        true
    }
}
