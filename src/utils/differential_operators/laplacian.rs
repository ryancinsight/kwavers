//! Laplacian operations module
//!
//! This module owns Laplacian computation knowledge following GRASP principles.
//! Single responsibility: Computing scalar field Laplacian.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, ArrayView3};
use num_traits::Float;
use super::coefficients::{FDCoefficients, SpatialOrder};

/// Compute Laplacian of a scalar field
///
/// Returns ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
///
/// # Arguments
/// * `field` - Input 3D scalar field
/// * `grid` - Grid information for spatial discretization
/// * `order` - Finite difference accuracy order
///
/// # Generic Parameters
/// * `T` - Float type (f32, f64) implementing num_traits::Float
///
/// # Errors
/// Returns error if field dimensions don't match grid.
pub fn laplacian<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<Array3<T>>
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

    let mut laplacian = Array3::<T>::zeros((nx, ny, nz));
    
    let pairs = FDCoefficients::second_derivative_pairs::<T>(order);
    let center = FDCoefficients::second_derivative_center::<T>(order);
    let stencil_radius = pairs.len();

    let dx2_inv = T::one() / (T::from(grid.dx).unwrap() * T::from(grid.dx).unwrap());
    let dy2_inv = T::one() / (T::from(grid.dy).unwrap() * T::from(grid.dy).unwrap());
    let dz2_inv = T::one() / (T::from(grid.dz).unwrap() * T::from(grid.dz).unwrap());

    // Compute Laplacian in interior points
    for i in stencil_radius..nx - stencil_radius {
        for j in stencil_radius..ny - stencil_radius {
            for k in stencil_radius..nz - stencil_radius {
                let mut d2f_dx2 = center * field[[i, j, k]];
                let mut d2f_dy2 = center * field[[i, j, k]];
                let mut d2f_dz2 = center * field[[i, j, k]];

                // Add symmetric pair contributions
                for (n, &coeff) in pairs.iter().enumerate() {
                    let offset = n + 1;
                    
                    // Second derivative in x
                    d2f_dx2 = d2f_dx2 + coeff * (field[[i + offset, j, k]] + field[[i - offset, j, k]]);
                    
                    // Second derivative in y
                    d2f_dy2 = d2f_dy2 + coeff * (field[[i, j + offset, k]] + field[[i, j - offset, k]]);
                    
                    // Second derivative in z
                    d2f_dz2 = d2f_dz2 + coeff * (field[[i, j, k + offset]] + field[[i, j, k - offset]]);
                }

                laplacian[[i, j, k]] = d2f_dx2 * dx2_inv + d2f_dy2 * dy2_inv + d2f_dz2 * dz2_inv;
            }
        }
    }

    Ok(laplacian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_laplacian_constant_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let field = Array3::<f64>::ones((5, 5, 5));

        let lap = laplacian(&field.view(), &grid, SpatialOrder::Second)?;

        // Laplacian of constant field should be zero
        assert_relative_eq!(lap[[2, 2, 2]], 0.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_laplacian_quadratic_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let mut field = Array3::<f64>::zeros((5, 5, 5));

        // Create quadratic field: f(x,y,z) = x² + y² + z²
        // Laplacian should be 2 + 2 + 2 = 6
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let x = i as f64;
                    let y = j as f64;
                    let z = k as f64;
                    field[[i, j, k]] = x * x + y * y + z * z;
                }
            }
        }

        let lap = laplacian(&field.view(), &grid, SpatialOrder::Second)?;

        // Check interior point
        assert_relative_eq!(lap[[2, 2, 2]], 6.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_laplacian_generic_types() -> KwaversResult<()> {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0)?;
        let field_f64 = Array3::<f64>::ones((3, 3, 3));
        let field_f32 = Array3::<f32>::ones((3, 3, 3));

        // Both should compile and run without errors
        let _ = laplacian(&field_f64.view(), &grid, SpatialOrder::Second)?;
        let _ = laplacian(&field_f32.view(), &grid, SpatialOrder::Second)?;

        Ok(())
    }
}