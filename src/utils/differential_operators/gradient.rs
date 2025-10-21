//! Gradient operations module
//!
//! This module owns gradient computation knowledge following GRASP principles.
//! Single responsibility: Computing spatial gradients with various accuracy orders.

use super::coefficients::{FDCoefficients, SpatialOrder};
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, ArrayView3};
use num_traits::Float;

/// Compute the gradient of a 3D field
///
/// This function implements central finite difference schemes with configurable
/// accuracy order, following the principle of parametric polymorphism for the
/// float type.
///
/// # Arguments
/// * `field` - Input 3D scalar field to differentiate
/// * `grid` - Grid information for spatial discretization
/// * `order` - Finite difference accuracy order
///
/// # Returns
/// * Tuple of (grad_x, grad_y, grad_z) arrays
///
/// # Generic Type Parameters
/// * `T` - Float type (f32, f64) implementing num_traits::Float
///
/// # Errors
/// Returns error if field dimensions don't match grid or if stencil extends
/// beyond domain boundaries.
///
/// # Example
/// ```rust
/// # use kwavers::utils::differential_operators::gradient;
/// # use kwavers::grid::Grid;
/// # use kwavers::utils::differential_operators::coefficients::SpatialOrder;
/// # use ndarray::Array3;
/// let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1)?;
/// let field = Array3::<f64>::zeros((10, 10, 10));
/// let (gx, gy, gz) = gradient(&field.view(), &grid, SpatialOrder::Second)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn gradient<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync,
{
    let shape = field.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Validate grid compatibility
    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(crate::error::KwaversError::Grid(
            crate::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    let mut grad_x = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_y = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_z = Array3::<T>::zeros((nx, ny, nz));

    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();

    // X-direction gradient
    let dx_inv = T::one() / T::from(grid.dx).unwrap();
    for i in stencil_radius..nx - stencil_radius {
        for j in 0..ny {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + coeff * (field[[i + offset, j, k]] - field[[i - offset, j, k]]);
                }
                grad_x[[i, j, k]] = grad_val * dx_inv;
            }
        }
    }

    // Y-direction gradient
    let dy_inv = T::one() / T::from(grid.dy).unwrap();
    for i in 0..nx {
        for j in stencil_radius..ny - stencil_radius {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + coeff * (field[[i, j + offset, k]] - field[[i, j - offset, k]]);
                }
                grad_y[[i, j, k]] = grad_val * dy_inv;
            }
        }
    }

    // Z-direction gradient
    let dz_inv = T::one() / T::from(grid.dz).unwrap();
    for i in 0..nx {
        for j in 0..ny {
            for k in stencil_radius..nz - stencil_radius {
                let mut grad_val = T::zero();
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + coeff * (field[[i, j, k + offset]] - field[[i, j, k - offset]]);
                }
                grad_z[[i, j, k]] = grad_val * dz_inv;
            }
        }
    }

    Ok((grad_x, grad_y, grad_z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_gradient_linear_function() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let mut field = Array3::<f64>::zeros((5, 5, 5));

        // Create linear function f(x,y,z) = 2x + 3y + z
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = 2.0 * i as f64 + 3.0 * j as f64 + k as f64;
                }
            }
        }

        let (grad_x, grad_y, grad_z) = gradient(&field.view(), &grid, SpatialOrder::Second)?;

        // Check interior points (gradient should be constant)
        assert_relative_eq!(grad_x[[2, 2, 2]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(grad_y[[2, 2, 2]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(grad_z[[2, 2, 2]], 1.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_gradient_generic_types() -> KwaversResult<()> {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0)?;
        let field_f64 = Array3::<f64>::ones((3, 3, 3));
        let field_f32 = Array3::<f32>::ones((3, 3, 3));

        // Both should compile and run without errors
        let _ = gradient(&field_f64.view(), &grid, SpatialOrder::Second)?;
        let _ = gradient(&field_f32.view(), &grid, SpatialOrder::Second)?;

        Ok(())
    }
}
