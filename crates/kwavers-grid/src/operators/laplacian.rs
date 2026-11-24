//! Laplacian operations module
//!
//! Unified Laplacian operator implementation for discretized grids.

use super::coefficients::{
    centered::{SecondDerivative, EIGHTH, FOURTH, SECOND, SIXTH},
    FdAccuracyOrder,
};
use crate::Grid;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3, ArrayViewMut3};
mod boundary;
mod interior;

/// Configuration for Laplacian computation
#[derive(Debug, Clone)]
pub struct LaplacianConfig {
    /// Finite difference order
    pub order: FdAccuracyOrder,
    /// Boundary condition type
    pub boundary: LaplacianBoundary,
}

/// Boundary condition for Laplacian operator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaplacianBoundary {
    /// Write zero wherever a centered stencil would extend outside the grid.
    Dirichlet,
    /// Enforce zero normal derivative by even reflection about each endpoint node.
    Neumann,
    /// Wrap samples around each axis at the selected stencil order.
    Periodic,
}

impl Default for LaplacianConfig {
    fn default() -> Self {
        Self {
            order: FdAccuracyOrder::Second,
            boundary: LaplacianBoundary::Dirichlet,
        }
    }
}

/// Unified Laplacian operator
#[derive(Debug)]
pub struct LaplacianOperator {
    config: LaplacianConfig,
    dx2_inv: f64,
    dy2_inv: f64,
    dz2_inv: f64,
}

impl LaplacianOperator {
    /// Create a new Laplacian operator
    pub fn new(grid: &Grid, config: LaplacianConfig) -> Self {
        Self {
            config,
            dx2_inv: 1.0 / (grid.dx * grid.dx),
            dy2_inv: 1.0 / (grid.dy * grid.dy),
            dz2_inv: 1.0 / (grid.dz * grid.dz),
        }
    }

    /// Create with default second-order accuracy
    pub fn second_order(grid: &Grid) -> Self {
        Self::new(grid, LaplacianConfig::default())
    }

    /// Create with specified order
    pub fn with_order(grid: &Grid, order: FdAccuracyOrder) -> Self {
        Self::new(
            grid,
            LaplacianConfig {
                order,
                boundary: LaplacianBoundary::Dirichlet,
            },
        )
    }

    /// Compute Laplacian of a scalar field
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn apply(&self, field: ArrayView3<'_, f64>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        let mut result = Array3::zeros([nx, ny, nz]);
        self.apply_mut(field, result.view_mut())?;
        Ok(result)
    }

    /// Write the Laplacian into a caller-owned, possibly strided view.
    /// Every output cell is overwritten. For spacing in metres, output units
    /// are input units per square metre.
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn apply_mut(
        &self,
        input: ArrayView3<f64>,
        output: ArrayViewMut3<f64>,
    ) -> KwaversResult<()> {
        if input.shape() != output.shape() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
                "Output dimensions {:?} don't match input {:?}",
                output.shape(),
                input.shape()
            )));
        }

        match self.config.order {
            FdAccuracyOrder::Second => self.apply_stencil(input, output, &SECOND),
            FdAccuracyOrder::Fourth => self.apply_stencil(input, output, &FOURTH),
            FdAccuracyOrder::Sixth => self.apply_stencil(input, output, &SIXTH),
            FdAccuracyOrder::Eighth => self.apply_stencil(input, output, &EIGHTH),
        }
        Ok(())
    }

    fn apply_stencil<const R: usize>(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        stencil: &SecondDerivative<R>,
    ) {
        if input.shape().contains(&0) {
            return;
        }
        if R == 1 {
            self.apply_second_order_interior(input, output.reborrow());
        } else {
            self.apply_interior(input, output.reborrow(), stencil);
        }
        self.apply_halo(input, output, stencil);
    }

    #[inline]
    fn value<const R: usize>(
        &self,
        stencil: &SecondDerivative<R>,
        sample: f64,
        pairs: impl Fn(usize) -> [f64; 3],
    ) -> f64 {
        let mut curvature = [stencil.center * sample; 3];
        for (n, &weight) in stencil.pairs.iter().enumerate() {
            let neighbors = pairs(n + 1);
            for axis in 0..3 {
                curvature[axis] += weight * neighbors[axis];
            }
        }
        curvature[2].mul_add(
            self.dz2_inv,
            curvature[0].mul_add(self.dx2_inv, curvature[1] * self.dy2_inv),
        )
    }
}

/// Compute Laplacian with specified order
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn laplacian(
    field: ArrayView3<f64>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<Array3<f64>> {
    let operator = LaplacianOperator::with_order(grid, order);
    operator.apply(field)
}

#[cfg(test)]
mod tests;
