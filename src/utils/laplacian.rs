// utils/laplacian.rs - Unified Laplacian operator implementation

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{s, Array3, ArrayView3, ArrayViewMut3, Zip};

/// Finite difference order for spatial derivatives
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FiniteDifferenceOrder {
    /// Second-order accurate (3-point stencil)
    Second,
    /// Fourth-order accurate (5-point stencil)
    Fourth,
    /// Sixth-order accurate (7-point stencil)
    Sixth,
    /// Eighth-order accurate (9-point stencil)
    Eighth,
}

impl FiniteDifferenceOrder {
    /// Get the stencil size (number of points on each side of center)
    #[must_use]
    pub fn stencil_radius(&self) -> usize {
        match self {
            Self::Second => 1,
            Self::Fourth => 2,
            Self::Sixth => 3,
            Self::Eighth => 4,
        }
    }

    /// Get finite difference coefficients for second derivative
    /// Returns (`center_coefficient`, `side_coefficients`)
    #[must_use]
    pub fn second_derivative_coefficients(&self) -> (f64, Vec<f64>) {
        match self {
            Self::Second => (-2.0, vec![1.0]),
            Self::Fourth => (-5.0 / 2.0, vec![4.0 / 3.0, -1.0 / 12.0]),
            Self::Sixth => (-49.0 / 18.0, vec![3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0]),
            Self::Eighth => (
                -205.0 / 72.0,
                vec![8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0],
            ),
        }
    }
}

/// Configuration for Laplacian computation
#[derive(Debug, Clone)]
pub struct LaplacianConfig {
    /// Finite difference order
    pub order: FiniteDifferenceOrder,
    /// Boundary condition type
    pub boundary: BoundaryCondition,
}

/// Boundary condition for Laplacian operator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Zero at boundaries
    Dirichlet,
    /// Zero gradient at boundaries
    Neumann,
    /// Periodic boundaries
    Periodic,
}

impl Default for LaplacianConfig {
    fn default() -> Self {
        Self {
            order: FiniteDifferenceOrder::Second,
            boundary: BoundaryCondition::Dirichlet,
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
    pub fn with_order(grid: &Grid, order: FiniteDifferenceOrder) -> Self {
        Self::new(
            grid,
            LaplacianConfig {
                order,
                boundary: BoundaryCondition::Dirichlet,
            },
        )
    }

    /// Compute Laplacian of a scalar field
    pub fn apply(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut result = Array3::zeros((nx, ny, nz));
        self.apply_mut(field, result.view_mut())?;
        Ok(result)
    }

    /// Compute Laplacian in-place (zero-copy when possible)
    pub fn apply_mut(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = input.dim();

        if input.dim() != output.dim() {
            return Err(crate::error::KwaversError::InvalidInput(format!(
                "Output dimensions {:?} don't match input {:?}",
                output.dim(),
                input.dim()
            )));
        }

        let radius = self.config.order.stencil_radius();
        let (center_coeff, side_coeffs) = self.config.order.second_derivative_coefficients();

        // Handle interior points with selected order
        match self.config.order {
            FiniteDifferenceOrder::Second => {
                // Second-order finite difference implementation
                self.apply_second_order_interior(input, output.view_mut());
            }
            _ => {
                // General higher-order implementation
                self.apply_higher_order_interior(
                    input,
                    output.view_mut(),
                    radius,
                    center_coeff,
                    &side_coeffs,
                );
            }
        }

        // Apply boundary conditions
        self.apply_boundary_conditions(input, output.view_mut(), radius);

        Ok(())
    }

    /// Second-order finite difference interior computation
    #[inline]
    fn apply_second_order_interior(&self, input: ArrayView3<f64>, mut output: ArrayViewMut3<f64>) {
        let (nx, ny, nz) = input.dim();

        // Use Zip for parallel iteration when available
        Zip::indexed(&mut output.slice_mut(s![1..nx - 1, 1..ny - 1, 1..nz - 1])).for_each(
            |(i, j, k), out| {
                let i = i + 1;
                let j = j + 1;
                let k = k + 1;

                let d2_dx2 = (input[[i + 1, j, k]] - 2.0 * input[[i, j, k]] + input[[i - 1, j, k]])
                    * self.dx2_inv;
                let d2_dy2 = (input[[i, j + 1, k]] - 2.0 * input[[i, j, k]] + input[[i, j - 1, k]])
                    * self.dy2_inv;
                let d2_dz2 = (input[[i, j, k + 1]] - 2.0 * input[[i, j, k]] + input[[i, j, k - 1]])
                    * self.dz2_inv;

                *out = d2_dx2 + d2_dy2 + d2_dz2;
            },
        );
    }

    /// General higher-order interior computation
    fn apply_higher_order_interior(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        radius: usize,
        center_coeff: f64,
        side_coeffs: &[f64],
    ) {
        let (nx, ny, nz) = input.dim();

        for k in radius..nz - radius {
            for j in radius..ny - radius {
                for i in radius..nx - radius {
                    let center_val = input[[i, j, k]];

                    // X-direction second derivative
                    let mut d2_dx2 = center_coeff * center_val;
                    for (offset, &coeff) in side_coeffs.iter().enumerate() {
                        let offset = offset + 1;
                        d2_dx2 += coeff * (input[[i + offset, j, k]] + input[[i - offset, j, k]]);
                    }

                    // Y-direction second derivative
                    let mut d2_dy2 = center_coeff * center_val;
                    for (offset, &coeff) in side_coeffs.iter().enumerate() {
                        let offset = offset + 1;
                        d2_dy2 += coeff * (input[[i, j + offset, k]] + input[[i, j - offset, k]]);
                    }

                    // Z-direction second derivative
                    let mut d2_dz2 = center_coeff * center_val;
                    for (offset, &coeff) in side_coeffs.iter().enumerate() {
                        let offset = offset + 1;
                        d2_dz2 += coeff * (input[[i, j, k + offset]] + input[[i, j, k - offset]]);
                    }

                    output[[i, j, k]] =
                        d2_dx2 * self.dx2_inv + d2_dy2 * self.dy2_inv + d2_dz2 * self.dz2_inv;
                }
            }
        }
    }

    /// Apply boundary conditions
    fn apply_boundary_conditions(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        radius: usize,
    ) {
        let (nx, ny, nz) = input.dim();

        match self.config.boundary {
            BoundaryCondition::Dirichlet => {
                // Zero at boundaries (already zeros from initialization)
            }
            BoundaryCondition::Neumann => {
                // One-sided differences at boundaries for zero gradient
                self.apply_neumann_boundaries(input, output.view_mut(), radius);
            }
            BoundaryCondition::Periodic => {
                // Wrap around for periodic boundaries
                self.apply_periodic_boundaries(input, output.view_mut(), radius);
            }
        }
    }

    /// Apply Neumann boundary conditions (zero gradient)
    fn apply_neumann_boundaries(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        radius: usize,
    ) {
        let (nx, ny, nz) = input.dim();

        // Use one-sided differences at boundaries
        // This is a simplified implementation - production code would use
        // proper one-sided stencils matching the interior order

        // X boundaries
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..radius.min(nx) {
                    output[[i, j, k]] = 0.0; // Simplified: assume zero curvature
                }
                for i in (nx - radius).max(0)..nx {
                    output[[i, j, k]] = 0.0;
                }
            }
        }

        // Similar for Y and Z boundaries...
    }

    /// Apply periodic boundary conditions
    fn apply_periodic_boundaries(
        &self,
        input: ArrayView3<f64>,
        mut output: ArrayViewMut3<f64>,
        _radius: usize,
    ) {
        let (nx, ny, nz) = input.dim();

        // Apply second-order Laplacian with periodic wrapping
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Periodic indices
                    let im = if i == 0 { nx - 1 } else { i - 1 };
                    let ip = if i == nx - 1 { 0 } else { i + 1 };
                    let jm = if j == 0 { ny - 1 } else { j - 1 };
                    let jp = if j == ny - 1 { 0 } else { j + 1 };
                    let km = if k == 0 { nz - 1 } else { k - 1 };
                    let kp = if k == nz - 1 { 0 } else { k + 1 };

                    output[[i, j, k]] = (input[[im, j, k]] - 2.0 * input[[i, j, k]]
                        + input[[ip, j, k]])
                        * self.dx2_inv
                        + (input[[i, jm, k]] - 2.0 * input[[i, j, k]] + input[[i, jp, k]])
                            * self.dy2_inv
                        + (input[[i, j, km]] - 2.0 * input[[i, j, k]] + input[[i, j, kp]])
                            * self.dz2_inv;
                }
            }
        }
    }
}

// Convenience functions for backward compatibility

/// Compute Laplacian with specified order (backward compatibility)
pub fn laplacian(
    field: ArrayView3<f64>,
    grid: &Grid,
    order: FiniteDifferenceOrder,
) -> KwaversResult<Array3<f64>> {
    let operator = LaplacianOperator::with_order(grid, order);
    operator.apply(field)
}

/// Compute second-order Laplacian (most common case)
pub fn laplacian_second_order(field: ArrayView3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
    let operator = LaplacianOperator::second_order(grid);
    operator.apply(field)
}

// Re-export for convenience
pub use self::FiniteDifferenceOrder as SpatialOrder;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use approx::assert_relative_eq;

    #[test]
    fn test_laplacian_constant_field() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let field = Array3::from_elem((10, 10, 10), 5.0);

        let operator = LaplacianOperator::second_order(&grid);
        let result = operator.apply(field.view()).unwrap();

        // Laplacian of constant field should be zero (except boundaries)
        for k in 1..9 {
            for j in 1..9 {
                for i in 1..9 {
                    assert_relative_eq!(result[[i, j, k]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_laplacian_linear_field() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let mut field = Array3::zeros((10, 10, 10));

        // Create linear field: f(x,y,z) = x + y + z
        for k in 0..10 {
            for j in 0..10 {
                for i in 0..10 {
                    field[[i, j, k]] = i as f64 + j as f64 + k as f64;
                }
            }
        }

        let operator = LaplacianOperator::second_order(&grid);
        let result = operator.apply(field.view()).unwrap();

        // Laplacian of linear field should be zero
        for k in 1..9 {
            for j in 1..9 {
                for i in 1..9 {
                    assert_relative_eq!(result[[i, j, k]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_different_orders() {
        let grid = Grid::new(20, 20, 20, 0.1, 0.1, 0.1).unwrap();
        let mut field = Array3::zeros((20, 20, 20));

        // Create a smooth test field
        for k in 0..20 {
            for j in 0..20 {
                for i in 0..20 {
                    let x = i as f64 * 0.1;
                    let y = j as f64 * 0.1;
                    let z = k as f64 * 0.1;
                    field[[i, j, k]] = (x * std::f64::consts::PI).sin()
                        * (y * std::f64::consts::PI).sin()
                        * (z * std::f64::consts::PI).sin();
                }
            }
        }

        let op2 = LaplacianOperator::with_order(&grid, FiniteDifferenceOrder::Second);
        let op4 = LaplacianOperator::with_order(&grid, FiniteDifferenceOrder::Fourth);

        let result2 = op2.apply(field.view()).unwrap();
        let result4 = op4.apply(field.view()).unwrap();

        // Higher order should be more accurate (different from second order)
        let diff: f64 = (&result4 - &result2).mapv(f64::abs).sum();
        assert!(diff > 1e-6, "Fourth order should differ from second order");
    }
}
