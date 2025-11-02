//! Finite difference operations for FDTD solver
//!
//! This module implements spatial derivatives using finite difference schemes
//! of various orders (2nd, 4th, 6th) for the FDTD method.

use crate::error::{KwaversError, KwaversResult};
use ndarray::{Array3, ArrayView3};
use std::collections::HashMap;

/// Finite difference operator
#[derive(Debug, Clone)]
pub struct FiniteDifference {
    /// Spatial order (2, 4, or 6)
    spatial_order: usize,
    /// Finite difference coefficients
    coefficients: HashMap<usize, Vec<f64>>,
}

impl FiniteDifference {
    /// Create a new finite difference operator
    pub fn new(spatial_order: usize) -> KwaversResult<Self> {
        if ![2, 4, 6].contains(&spatial_order) {
            return Err(KwaversError::InvalidInput(format!(
                "spatial_order must be 2, 4, or 6, got {spatial_order}"
            )));
        }

        let mut coefficients = HashMap::new();

        // Central difference coefficients for first derivative
        // These are the coefficients for (f(x+ih) - f(x-ih)) terms
        coefficients.insert(2, vec![0.5]); // 2nd order: 1/(2h) * [f(x+h) - f(x-h)]
        coefficients.insert(4, vec![2.0 / 3.0, -1.0 / 12.0]); // 4th order
        coefficients.insert(6, vec![3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0]); // 6th order

        Ok(Self {
            spatial_order,
            coefficients,
        })
    }

    /// Get coefficients for the current spatial order
    #[must_use]
    pub fn get_coefficients(&self) -> &Vec<f64> {
        &self.coefficients[&self.spatial_order]
    }

    /// Compute spatial derivative along a given axis
    pub fn compute_derivative(
        &self,
        field: &ArrayView3<f64>,
        axis: usize,
        spacing: f64,
    ) -> KwaversResult<Array3<f64>> {
        if axis > 2 {
            return Err(KwaversError::InvalidInput(format!(
                "axis must be 0, 1, or 2, got {axis}"
            )));
        }

        let (nx, ny, nz) = field.dim();
        let mut deriv = Array3::zeros((nx, ny, nz));
        let coeffs = self.get_coefficients();
        let n_coeffs = coeffs.len();

        // Determine bounds based on stencil size
        let half_stencil = n_coeffs;
        let start = half_stencil;
        let (end_x, end_y, end_z) = (nx - half_stencil, ny - half_stencil, nz - half_stencil);

        // Apply finite differences in the interior with optimal cache locality
        match axis {
            0 => {
                // X-direction differentiation: iterate over i (contiguous) first
                for k in start..end_z {
                    for j in start..end_y {
                        for i in start..end_x {
                            let mut val = 0.0;

                            // Apply stencil coefficients along x-direction
                            for (idx, &coeff) in coeffs.iter().enumerate() {
                                let offset = idx + 1;
                                val += coeff * (field[[i + offset, j, k]] - field[[i - offset, j, k]]);
                            }

                            deriv[[i, j, k]] = val / spacing;
                        }
                    }
                }
            }
            1 => {
                // Y-direction differentiation: iterate over j first for better cache locality
                for k in start..end_z {
                    for i in start..end_x {
                        for j in start..end_y {
                            let mut val = 0.0;

                            // Apply stencil coefficients along y-direction
                            for (idx, &coeff) in coeffs.iter().enumerate() {
                                let offset = idx + 1;
                                val += coeff * (field[[i, j + offset, k]] - field[[i, j - offset, k]]);
                            }

                            deriv[[i, j, k]] = val / spacing;
                        }
                    }
                }
            }
            2 => {
                // Z-direction differentiation: iterate over k first for better cache locality
                for j in start..end_y {
                    for i in start..end_x {
                        for k in start..end_z {
                            let mut val = 0.0;

                            // Apply stencil coefficients along z-direction
                            for (idx, &coeff) in coeffs.iter().enumerate() {
                                let offset = idx + 1;
                                val += coeff * (field[[i, j, k + offset]] - field[[i, j, k - offset]]);
                            }

                            deriv[[i, j, k]] = val / spacing;
                        }
                    }
                }
            }
            _ => {
                return Err(crate::KwaversError::Config(
                    crate::ConfigError::InvalidValue {
                        parameter: "axis".to_string(),
                        value: axis.to_string(),
                        constraint: "0, 1, or 2".to_string(),
                    },
                ));
            }
        }

        // Apply boundary derivatives
        self.apply_boundary_derivatives(&mut deriv, field, axis, spacing, nx, ny, nz)?;

        Ok(deriv)
    }

    /// Apply lower-order derivatives at boundaries
    fn apply_boundary_derivatives(
        &self,
        deriv: &mut Array3<f64>,
        field: &ArrayView3<f64>,
        axis: usize,
        spacing: f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // Use second-order forward/backward differences at boundaries
        let forward_coeffs = [-1.5, 2.0, -0.5];
        let backward_coeffs = [0.5, -2.0, 1.5];

        match axis {
            0 => {
                // Left boundary (i = 0)
                for j in 0..ny {
                    for k in 0..nz {
                        deriv[[0, j, k]] = (forward_coeffs[0] * field[[0, j, k]]
                            + forward_coeffs[1] * field[[1, j, k]]
                            + forward_coeffs[2] * field[[2, j, k]])
                            / spacing;
                    }
                }
                // Right boundary (i = nx-1)
                for j in 0..ny {
                    for k in 0..nz {
                        deriv[[nx - 1, j, k]] = (backward_coeffs[0] * field[[nx - 3, j, k]]
                            + backward_coeffs[1] * field[[nx - 2, j, k]]
                            + backward_coeffs[2] * field[[nx - 1, j, k]])
                            / spacing;
                    }
                }
            }
            1 => {
                // Bottom boundary (j = 0)
                for i in 0..nx {
                    for k in 0..nz {
                        deriv[[i, 0, k]] = (forward_coeffs[0] * field[[i, 0, k]]
                            + forward_coeffs[1] * field[[i, 1, k]]
                            + forward_coeffs[2] * field[[i, 2, k]])
                            / spacing;
                    }
                }
                // Top boundary (j = ny-1)
                for i in 0..nx {
                    for k in 0..nz {
                        deriv[[i, ny - 1, k]] = (backward_coeffs[0] * field[[i, ny - 3, k]]
                            + backward_coeffs[1] * field[[i, ny - 2, k]]
                            + backward_coeffs[2] * field[[i, ny - 1, k]])
                            / spacing;
                    }
                }
            }
            2 => {
                // Front boundary (k = 0)
                for i in 0..nx {
                    for j in 0..ny {
                        deriv[[i, j, 0]] = (forward_coeffs[0] * field[[i, j, 0]]
                            + forward_coeffs[1] * field[[i, j, 1]]
                            + forward_coeffs[2] * field[[i, j, 2]])
                            / spacing;
                    }
                }
                // Back boundary (k = nz-1)
                for i in 0..nx {
                    for j in 0..ny {
                        deriv[[i, j, nz - 1]] = (backward_coeffs[0] * field[[i, j, nz - 3]]
                            + backward_coeffs[1] * field[[i, j, nz - 2]]
                            + backward_coeffs[2] * field[[i, j, nz - 1]])
                            / spacing;
                    }
                }
            }
            _ => {
                return Err(crate::KwaversError::Config(
                    crate::ConfigError::InvalidValue {
                        parameter: "axis".to_string(),
                        value: axis.to_string(),
                        constraint: "0, 1, or 2".to_string(),
                    },
                ));
            }
        }

        Ok(())
    }

    /// Compute divergence of a vector field
    pub fn compute_divergence(
        &self,
        vx: &ArrayView3<f64>,
        vy: &ArrayView3<f64>,
        vz: &ArrayView3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Array3<f64>> {
        let dvx_dx = self.compute_derivative(vx, 0, dx)?;
        let dvy_dy = self.compute_derivative(vy, 1, dy)?;
        let dvz_dz = self.compute_derivative(vz, 2, dz)?;

        Ok(&dvx_dx + &dvy_dy + &dvz_dz)
    }

    /// Compute gradient of a scalar field
    pub fn compute_gradient(
        &self,
        field: &ArrayView3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        let grad_x = self.compute_derivative(field, 0, dx)?;
        let grad_y = self.compute_derivative(field, 1, dy)?;
        let grad_z = self.compute_derivative(field, 2, dz)?;

        Ok((grad_x, grad_y, grad_z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_difference_creation() {
        let fd = FiniteDifference::new(2);
        assert!(fd.is_ok());

        let fd = FiniteDifference::new(4);
        assert!(fd.is_ok());

        let fd = FiniteDifference::new(6);
        assert!(fd.is_ok());

        let fd = FiniteDifference::new(3);
        assert!(fd.is_err());
    }

    #[test]
    fn test_derivative_linear_field() {
        let fd = FiniteDifference::new(2).unwrap();

        // Create a linear field (derivative should be constant)
        let mut field = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    field[[i, j, k]] = i as f64; // Linear in x
                }
            }
        }

        let deriv = fd.compute_derivative(&field.view(), 0, 1.0).unwrap();

        // Check that derivative is approximately 1.0 in the interior
        for i in 1..9 {
            for j in 1..9 {
                for k in 1..9 {
                    assert!(
                        (deriv[[i, j, k]] - 1.0).abs() < 1e-10,
                        "Expected derivative 1.0, got {}",
                        deriv[[i, j, k]]
                    );
                }
            }
        }
    }
}
