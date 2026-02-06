//! Numerical operators for nonlinear elastic wave propagation
//!
//! This module implements finite difference operators for computing spatial
//! derivatives and differential operators on structured 3D grids.
//!
//! ## Implemented Operators
//!
//! 1. **Laplacian**: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
//! 2. **Divergence of gradient product**: ∇·(∇u₁ ⊗ ∇u₂)
//!
//! ## Numerical Methods
//!
//! All operators use second-order accurate central differences:
//! - ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / Δx²
//! - ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2Δx)
//!
//! Boundary conditions are handled by returning zero for points near the boundary,
//! which is appropriate for the interior-focused elastography applications.
//!
//! ## Literature References
//!
//! - LeVeque, R. J. (2007). "Finite Difference Methods for Ordinary and Partial
//!   Differential Equations", SIAM.
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids", Mathematics of Computation, 51(184), 699-706.

use crate::domain::grid::Grid;
use ndarray::Array3;

/// Numerical operators for nonlinear wave propagation
///
/// This structure provides methods for computing spatial derivatives and
/// differential operators needed for nonlinear elastic wave equations.
#[derive(Debug, Clone)]
pub struct NumericsOperators {
    /// Computational grid
    grid: Grid,
}

impl NumericsOperators {
    /// Create new numerics operators for the given grid
    #[must_use]
    pub fn new(grid: Grid) -> Self {
        Self { grid }
    }

    /// Compute Laplacian ∇²u at a grid point
    ///
    /// # Theorem Reference
    /// The Laplacian operator in Cartesian coordinates is:
    /// ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
    ///
    /// Using second-order central finite differences:
    /// ∂²u/∂x² ≈ (u[i+1,j,k] - 2u[i,j,k] + u[i-1,j,k]) / Δx²
    ///
    /// Truncation error: O(Δx²) for smooth solutions
    ///
    /// # Arguments
    /// * `i` - x grid index
    /// * `j` - y grid index
    /// * `k` - z grid index
    /// * `u` - 3D displacement field
    ///
    /// # Returns
    /// Laplacian value at grid point (i,j,k) in units of 1/m²
    ///
    /// # Boundary Conditions
    /// Returns 0.0 for points on or near the boundary (within 1 grid point)
    #[must_use]
    pub fn laplacian(&self, i: usize, j: usize, k: usize, u: &Array3<f64>) -> f64 {
        let dx2 = self.grid.dx * self.grid.dx;
        let dy2 = self.grid.dy * self.grid.dy;
        let dz2 = self.grid.dz * self.grid.dz;

        let d2u_dx2 = if i > 0 && i < self.grid.nx - 1 {
            (u[[i + 1, j, k]] - 2.0 * u[[i, j, k]] + u[[i - 1, j, k]]) / dx2
        } else {
            0.0
        };

        let d2u_dy2 = if j > 0 && j < self.grid.ny - 1 {
            (u[[i, j + 1, k]] - 2.0 * u[[i, j, k]] + u[[i, j - 1, k]]) / dy2
        } else {
            0.0
        };

        let d2u_dz2 = if k > 0 && k < self.grid.nz - 1 {
            (u[[i, j, k + 1]] - 2.0 * u[[i, j, k]] + u[[i, j, k - 1]]) / dz2
        } else {
            0.0
        };

        d2u_dx2 + d2u_dy2 + d2u_dz2
    }

    /// Compute divergence of gradient product ∇·(∇u₁ ⊗ ∇u₂) for harmonic generation
    ///
    /// # Theorem Reference
    /// Chen (2013): Third harmonic generation involves term 2∇u₁·∇u₂
    /// This is implemented as ∇·(∇u₁ ⊗ ∇u₂) for numerical stability.
    ///
    /// In component form:
    /// ∇·(∇u₁ ⊗ ∇u₂) = ∂/∂x(∂u₁/∂x * ∂u₂/∂x) + ∂/∂y(∂u₁/∂y * ∂u₂/∂y) + ∂/∂z(∂u₁/∂z * ∂u₂/∂z)
    ///
    /// Reference: Chen, S., et al. (2013). "Harmonic motion detection in ultrasound elastography."
    /// IEEE Transactions on Medical Imaging, 32(5), 863-874.
    ///
    /// # Arguments
    /// * `i` - x grid index
    /// * `j` - y grid index
    /// * `k` - z grid index
    /// * `u1` - First displacement field
    /// * `u2` - Second displacement field
    ///
    /// # Returns
    /// Divergence value at grid point (i,j,k) in units of 1/m²
    ///
    /// # Boundary Conditions
    /// Returns 0.0 for points on or within 2 grid points of boundary (stencil requirement)
    #[must_use]
    pub fn divergence_product(
        &self,
        i: usize,
        j: usize,
        k: usize,
        u1: &Array3<f64>,
        u2: &Array3<f64>,
    ) -> f64 {
        // Compute ∇·(∇u₁ ⊗ ∇u₂) = ∂/∂x(∂u₁/∂x * ∂u₂/∂x) + ∂/∂y(∂u₁/∂y * ∂u₂/∂y) + ∂/∂z(∂u₁/∂z * ∂u₂/∂z)
        // This requires computing derivatives of the product of gradients at neighboring points

        if i < 2
            || i >= self.grid.nx - 2
            || j < 2
            || j >= self.grid.ny - 2
            || k < 2
            || k >= self.grid.nz - 2
        {
            return 0.0;
        }

        // Compute ∂/∂x(∂u₁/∂x * ∂u₂/∂x) using central differences
        let du1_dx_ip1 = (u1[[i + 2, j, k]] - u1[[i, j, k]]) / (2.0 * self.grid.dx);
        let du2_dx_ip1 = (u2[[i + 2, j, k]] - u2[[i, j, k]]) / (2.0 * self.grid.dx);
        let product_ip1 = du1_dx_ip1 * du2_dx_ip1;

        let du1_dx_im1 = (u1[[i, j, k]] - u1[[i - 2, j, k]]) / (2.0 * self.grid.dx);
        let du2_dx_im1 = (u2[[i, j, k]] - u2[[i - 2, j, k]]) / (2.0 * self.grid.dx);
        let product_im1 = du1_dx_im1 * du2_dx_im1;

        let d_dx = (product_ip1 - product_im1) / (2.0 * self.grid.dx);

        // Compute ∂/∂y(∂u₁/∂y * ∂u₂/∂y)
        let du1_dy_jp1 = (u1[[i, j + 2, k]] - u1[[i, j, k]]) / (2.0 * self.grid.dy);
        let du2_dy_jp1 = (u2[[i, j + 2, k]] - u2[[i, j, k]]) / (2.0 * self.grid.dy);
        let product_jp1 = du1_dy_jp1 * du2_dy_jp1;

        let du1_dy_jm1 = (u1[[i, j, k]] - u1[[i, j - 2, k]]) / (2.0 * self.grid.dy);
        let du2_dy_jm1 = (u2[[i, j, k]] - u2[[i, j - 2, k]]) / (2.0 * self.grid.dy);
        let product_jm1 = du1_dy_jm1 * du2_dy_jm1;

        let d_dy = (product_jp1 - product_jm1) / (2.0 * self.grid.dy);

        // Compute ∂/∂z(∂u₁/∂z * ∂u₂/∂z)
        let du1_dz_kp1 = (u1[[i, j, k + 2]] - u1[[i, j, k]]) / (2.0 * self.grid.dz);
        let du2_dz_kp1 = (u2[[i, j, k + 2]] - u2[[i, j, k]]) / (2.0 * self.grid.dz);
        let product_kp1 = du1_dz_kp1 * du2_dz_kp1;

        let du1_dz_km1 = (u1[[i, j, k]] - u1[[i, j, k - 2]]) / (2.0 * self.grid.dz);
        let du2_dz_km1 = (u2[[i, j, k]] - u2[[i, j, k - 2]]) / (2.0 * self.grid.dz);
        let product_km1 = du1_dz_km1 * du2_dz_km1;

        let d_dz = (product_kp1 - product_km1) / (2.0 * self.grid.dz);

        d_dx + d_dy + d_dz
    }

    /// Compute gradient at a grid point
    ///
    /// # Arguments
    /// * `i` - x grid index
    /// * `j` - y grid index
    /// * `k` - z grid index
    /// * `u` - Scalar field
    ///
    /// # Returns
    /// Gradient vector [∂u/∂x, ∂u/∂y, ∂u/∂z]
    #[must_use]
    pub fn gradient(&self, i: usize, j: usize, k: usize, u: &Array3<f64>) -> [f64; 3] {
        let du_dx = if i > 0 && i < self.grid.nx - 1 {
            (u[[i + 1, j, k]] - u[[i - 1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            0.0
        };

        let du_dy = if j > 0 && j < self.grid.ny - 1 {
            (u[[i, j + 1, k]] - u[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };

        let du_dz = if k > 0 && k < self.grid.nz - 1 {
            (u[[i, j, k + 1]] - u[[i, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            0.0
        };

        [du_dx, du_dy, du_dz]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_constant_field() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let ops = NumericsOperators::new(grid);

        let u = Array3::from_elem((10, 10, 10), 1.0);
        let lap = ops.laplacian(5, 5, 5, &u);

        // Laplacian of constant field should be zero
        assert!((lap - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_linear_field() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let ops = NumericsOperators::new(grid);

        let mut u = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    u[[i, j, k]] = i as f64 * 0.001; // Linear in x
                }
            }
        }

        let lap = ops.laplacian(5, 5, 5, &u);

        // Laplacian of linear field should be zero
        assert!((lap - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_quadratic_field() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0).unwrap();
        let ops = NumericsOperators::new(grid);

        let mut u = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let x = i as f64;
                    u[[i, j, k]] = x * x; // u = x²
                }
            }
        }

        let lap = ops.laplacian(5, 5, 5, &u);

        // Laplacian of x² should be 2
        assert!((lap - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_boundary() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let ops = NumericsOperators::new(grid);

        let u = Array3::from_elem((10, 10, 10), 1.0);

        // Boundary points should return 0
        let lap_boundary = ops.laplacian(0, 5, 5, &u);
        assert_eq!(lap_boundary, 0.0);

        let lap_boundary_end = ops.laplacian(9, 5, 5, &u);
        assert_eq!(lap_boundary_end, 0.0);
    }

    #[test]
    fn test_gradient_linear_field() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0).unwrap();
        let ops = NumericsOperators::new(grid);

        let mut u = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    u[[i, j, k]] = 2.0 * i as f64 + 3.0 * j as f64 + 4.0 * k as f64;
                }
            }
        }

        let grad = ops.gradient(5, 5, 5, &u);

        // Gradient should be [2, 3, 4]
        assert!((grad[0] - 2.0).abs() < 1e-10);
        assert!((grad[1] - 3.0).abs() < 1e-10);
        assert!((grad[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_divergence_product_zero_fields() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let ops = NumericsOperators::new(grid);

        let u1 = Array3::zeros((10, 10, 10));
        let u2 = Array3::zeros((10, 10, 10));

        let div = ops.divergence_product(5, 5, 5, &u1, &u2);
        assert_eq!(div, 0.0);
    }

    #[test]
    fn test_divergence_product_boundary() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let ops = NumericsOperators::new(grid);

        let u1 = Array3::from_elem((10, 10, 10), 1.0);
        let u2 = Array3::from_elem((10, 10, 10), 1.0);

        // Points within 2 grid points of boundary should return 0
        let div_boundary = ops.divergence_product(1, 5, 5, &u1, &u2);
        assert_eq!(div_boundary, 0.0);

        let div_boundary_end = ops.divergence_product(8, 5, 5, &u1, &u2);
        assert_eq!(div_boundary_end, 0.0);
    }

    #[test]
    fn test_numerics_operators_new() {
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
        let ops = NumericsOperators::new(grid.clone());

        assert_eq!(ops.grid.nx, 16);
        assert_eq!(ops.grid.ny, 16);
        assert_eq!(ops.grid.nz, 16);
    }
}
