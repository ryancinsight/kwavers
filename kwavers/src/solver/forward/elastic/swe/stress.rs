//! Stress tensor derivative computations for elastic wave propagation
//!
//! Implements fourth-order accurate finite difference stencils for computing
//! spatial derivatives of the stress tensor components.
//!
//! ## Mathematical Background
//!
//! The elastic wave equation in terms of stress and velocity:
//! ```text
//! ρ ∂v/∂t = ∇·σ
//! ```
//!
//! Where σ is the stress tensor:
//! ```text
//! σ = [σxx  σxy  σxz]
//!     [σyx  σyy  σyz]
//!     [σzx  σzy  σzz]
//! ```
//!
//! The divergence of stress provides the acceleration:
//! ```text
//! ∇·σ = [∂σxx/∂x + ∂σxy/∂y + ∂σxz/∂z]
//!       [∂σyx/∂x + ∂σyy/∂y + ∂σyz/∂z]
//!       [∂σzx/∂x + ∂σzy/∂y + ∂σzz/∂z]
//! ```
//!
//! ## Numerical Method
//!
//! Fourth-order accurate centered finite difference:
//! ```text
//! ∂f/∂x ≈ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12·Δx)
//! ```
//!
//! ## References
//!
//! - Moczo, P., et al. (2007). "3D finite-difference method for elastodynamics."
//!   Solid Earth, 93(3), 523-553.

use super::types::ElasticWaveField;
use crate::domain::grid::Grid;

/// Stress tensor derivative calculator
///
/// Computes spatial derivatives of stress tensor components using
/// fourth-order accurate finite difference stencils.
#[derive(Debug)]
pub struct StressDerivatives<'a> {
    grid: &'a Grid,
}

impl<'a> StressDerivatives<'a> {
    /// Create new stress derivative calculator
    #[must_use]
    pub fn new(grid: &'a Grid) -> Self {
        Self { grid }
    }

    /// Compute ∂σxx/∂x using fourth-order finite differences
    ///
    /// ## Arguments
    /// - `i, j, k`: Grid indices
    /// - `field`: Current wave field state
    ///
    /// ## Returns
    /// Derivative value in Pa/m
    ///
    /// ## Boundary Treatment
    /// Uses second-order one-sided stencils near boundaries (i < 2 or i >= nx-2)
    #[must_use]
    pub fn stress_xx_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (nx, _ny, _nz) = field.ux.dim();
        let dx = self.grid.dx;

        if i < 2 || i >= nx - 2 {
            // Second-order one-sided stencil at boundaries
            if i == 0 {
                (field.ux[[i + 1, j, k]] - field.ux[[i, j, k]]) / dx
            } else if i == 1 || i == nx - 2 {
                (field.ux[[i + 1, j, k]] - field.ux[[i - 1, j, k]]) / (2.0 * dx)
            } else {
                (field.ux[[i, j, k]] - field.ux[[i - 1, j, k]]) / dx
            }
        } else {
            // Fourth-order centered stencil (interior)
            (-field.ux[[i + 2, j, k]] + 8.0 * field.ux[[i + 1, j, k]]
                - 8.0 * field.ux[[i - 1, j, k]]
                + field.ux[[i - 2, j, k]])
                / (12.0 * dx)
        }
    }

    /// Compute ∂σxy/∂y using fourth-order finite differences
    #[must_use]
    pub fn stress_xy_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (_nx, ny, _nz) = field.ux.dim();
        let dy = self.grid.dy;

        if j < 2 || j >= ny - 2 {
            // Second-order stencil at boundaries
            if j == 0 {
                (field.ux[[i, j + 1, k]] - field.ux[[i, j, k]]) / dy
            } else if j == 1 || j == ny - 2 {
                (field.ux[[i, j + 1, k]] - field.ux[[i, j - 1, k]]) / (2.0 * dy)
            } else {
                (field.ux[[i, j, k]] - field.ux[[i, j - 1, k]]) / dy
            }
        } else {
            // Fourth-order centered stencil
            (-field.ux[[i, j + 2, k]] + 8.0 * field.ux[[i, j + 1, k]]
                - 8.0 * field.ux[[i, j - 1, k]]
                + field.ux[[i, j - 2, k]])
                / (12.0 * dy)
        }
    }

    /// Compute ∂σxz/∂z using fourth-order finite differences
    #[must_use]
    pub fn stress_xz_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (_nx, _ny, nz) = field.ux.dim();
        let dz = self.grid.dz;

        if k < 2 || k >= nz - 2 {
            // Second-order stencil at boundaries
            if k == 0 {
                (field.ux[[i, j, k + 1]] - field.ux[[i, j, k]]) / dz
            } else if k == 1 || k == nz - 2 {
                (field.ux[[i, j, k + 1]] - field.ux[[i, j, k - 1]]) / (2.0 * dz)
            } else {
                (field.ux[[i, j, k]] - field.ux[[i, j, k - 1]]) / dz
            }
        } else {
            // Fourth-order centered stencil
            (-field.ux[[i, j, k + 2]] + 8.0 * field.ux[[i, j, k + 1]]
                - 8.0 * field.ux[[i, j, k - 1]]
                + field.ux[[i, j, k - 2]])
                / (12.0 * dz)
        }
    }

    /// Compute ∂σyx/∂x (same as ∂σxy/∂x by symmetry)
    #[must_use]
    pub fn stress_yx_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (nx, _ny, _nz) = field.uy.dim();
        let dx = self.grid.dx;

        if i < 2 || i >= nx - 2 {
            if i == 0 {
                (field.uy[[i + 1, j, k]] - field.uy[[i, j, k]]) / dx
            } else if i == 1 || i == nx - 2 {
                (field.uy[[i + 1, j, k]] - field.uy[[i - 1, j, k]]) / (2.0 * dx)
            } else {
                (field.uy[[i, j, k]] - field.uy[[i - 1, j, k]]) / dx
            }
        } else {
            (-field.uy[[i + 2, j, k]] + 8.0 * field.uy[[i + 1, j, k]]
                - 8.0 * field.uy[[i - 1, j, k]]
                + field.uy[[i - 2, j, k]])
                / (12.0 * dx)
        }
    }

    /// Compute ∂σyy/∂y
    #[must_use]
    pub fn stress_yy_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (_nx, ny, _nz) = field.uy.dim();
        let dy = self.grid.dy;

        if j < 2 || j >= ny - 2 {
            if j == 0 {
                (field.uy[[i, j + 1, k]] - field.uy[[i, j, k]]) / dy
            } else if j == 1 || j == ny - 2 {
                (field.uy[[i, j + 1, k]] - field.uy[[i, j - 1, k]]) / (2.0 * dy)
            } else {
                (field.uy[[i, j, k]] - field.uy[[i, j - 1, k]]) / dy
            }
        } else {
            (-field.uy[[i, j + 2, k]] + 8.0 * field.uy[[i, j + 1, k]]
                - 8.0 * field.uy[[i, j - 1, k]]
                + field.uy[[i, j - 2, k]])
                / (12.0 * dy)
        }
    }

    /// Compute ∂σyz/∂z
    #[must_use]
    pub fn stress_yz_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (_nx, _ny, nz) = field.uy.dim();
        let dz = self.grid.dz;

        if k < 2 || k >= nz - 2 {
            if k == 0 {
                (field.uy[[i, j, k + 1]] - field.uy[[i, j, k]]) / dz
            } else if k == 1 || k == nz - 2 {
                (field.uy[[i, j, k + 1]] - field.uy[[i, j, k - 1]]) / (2.0 * dz)
            } else {
                (field.uy[[i, j, k]] - field.uy[[i, j, k - 1]]) / dz
            }
        } else {
            (-field.uy[[i, j, k + 2]] + 8.0 * field.uy[[i, j, k + 1]]
                - 8.0 * field.uy[[i, j, k - 1]]
                + field.uy[[i, j, k - 2]])
                / (12.0 * dz)
        }
    }

    /// Compute ∂σzx/∂x
    #[must_use]
    pub fn stress_zx_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (nx, _ny, _nz) = field.uz.dim();
        let dx = self.grid.dx;

        if i < 2 || i >= nx - 2 {
            if i == 0 {
                (field.uz[[i + 1, j, k]] - field.uz[[i, j, k]]) / dx
            } else if i == 1 || i == nx - 2 {
                (field.uz[[i + 1, j, k]] - field.uz[[i - 1, j, k]]) / (2.0 * dx)
            } else {
                (field.uz[[i, j, k]] - field.uz[[i - 1, j, k]]) / dx
            }
        } else {
            (-field.uz[[i + 2, j, k]] + 8.0 * field.uz[[i + 1, j, k]]
                - 8.0 * field.uz[[i - 1, j, k]]
                + field.uz[[i - 2, j, k]])
                / (12.0 * dx)
        }
    }

    /// Compute ∂σzy/∂y
    #[must_use]
    pub fn stress_zy_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (_nx, ny, _nz) = field.uz.dim();
        let dy = self.grid.dy;

        if j < 2 || j >= ny - 2 {
            if j == 0 {
                (field.uz[[i, j + 1, k]] - field.uz[[i, j, k]]) / dy
            } else if j == 1 || j == ny - 2 {
                (field.uz[[i, j + 1, k]] - field.uz[[i, j - 1, k]]) / (2.0 * dy)
            } else {
                (field.uz[[i, j, k]] - field.uz[[i, j - 1, k]]) / dy
            }
        } else {
            (-field.uz[[i, j + 2, k]] + 8.0 * field.uz[[i, j + 1, k]]
                - 8.0 * field.uz[[i, j - 1, k]]
                + field.uz[[i, j - 2, k]])
                / (12.0 * dy)
        }
    }

    /// Compute ∂σzz/∂z
    #[must_use]
    pub fn stress_zz_derivative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> f64 {
        let (_nx, _ny, nz) = field.uz.dim();
        let dz = self.grid.dz;

        if k < 2 || k >= nz - 2 {
            if k == 0 {
                (field.uz[[i, j, k + 1]] - field.uz[[i, j, k]]) / dz
            } else if k == 1 || k == nz - 2 {
                (field.uz[[i, j, k + 1]] - field.uz[[i, j, k - 1]]) / (2.0 * dz)
            } else {
                (field.uz[[i, j, k]] - field.uz[[i, j, k - 1]]) / dz
            }
        } else {
            (-field.uz[[i, j, k + 2]] + 8.0 * field.uz[[i, j, k + 1]]
                - 8.0 * field.uz[[i, j, k - 1]]
                + field.uz[[i, j, k - 2]])
                / (12.0 * dz)
        }
    }

    /// Compute full stress divergence at a point
    ///
    /// Returns: `[∂σxx/∂x + ∂σxy/∂y + ∂σxz/∂z, ...]` for all three components
    #[must_use]
    pub fn stress_divergence(
        &self,
        i: usize,
        j: usize,
        k: usize,
        field: &ElasticWaveField,
    ) -> [f64; 3] {
        let div_x = self.stress_xx_derivative(i, j, k, field)
            + self.stress_xy_derivative(i, j, k, field)
            + self.stress_xz_derivative(i, j, k, field);

        let div_y = self.stress_yx_derivative(i, j, k, field)
            + self.stress_yy_derivative(i, j, k, field)
            + self.stress_yz_derivative(i, j, k, field);

        let div_z = self.stress_zx_derivative(i, j, k, field)
            + self.stress_zy_derivative(i, j, k, field)
            + self.stress_zz_derivative(i, j, k, field);

        [div_x, div_y, div_z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_stress_derivatives_basic() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let stress_calc = StressDerivatives::new(&grid);
        let field = ElasticWaveField::new(10, 10, 10);

        // Zero field should give zero derivatives
        let deriv = stress_calc.stress_xx_derivative(5, 5, 5, &field);
        assert!((deriv).abs() < 1e-10);
    }

    #[test]
    fn test_stress_divergence() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let stress_calc = StressDerivatives::new(&grid);
        let field = ElasticWaveField::new(10, 10, 10);

        let div = stress_calc.stress_divergence(5, 5, 5, &field);
        assert!(div[0].abs() < 1e-10);
        assert!(div[1].abs() < 1e-10);
        assert!(div[2].abs() < 1e-10);
    }
}
