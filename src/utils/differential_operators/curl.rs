//! Curl operations module
//!
//! This module owns curl computation knowledge following GRASP principles.
//! Single responsibility: Computing vector field curl.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, ArrayView3};
use num_traits::Float;
use super::coefficients::{FDCoefficients, SpatialOrder};

/// Compute curl of a vector field
///
/// Returns ∇×v = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)
///
/// # Arguments
/// * `vx`, `vy`, `vz` - Components of the vector field
/// * `grid` - Grid information for spatial discretization
/// * `order` - Finite difference accuracy order
///
/// # Generic Parameters
/// * `T` - Float type (f32, f64) implementing num_traits::Float
///
/// # Returns
/// Tuple of (curl_x, curl_y, curl_z) arrays
///
/// # Errors
/// Returns error if field dimensions don't match grid.
pub fn curl<T>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync,
{
    let shape = vx.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Validate grid compatibility and vector field consistency
    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(crate::error::KwaversError::Grid(
            crate::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    if vy.shape() != shape || vz.shape() != shape {
        return Err(crate::error::KwaversError::Grid(
            crate::error::GridError::DimensionMismatch {
                expected: format!("Vector field components must have same dimensions"),
                actual: format!("vx: {:?}, vy: {:?}, vz: {:?}", vx.shape(), vy.shape(), vz.shape()),
            },
        ));
    }

    let mut curl_x = Array3::<T>::zeros((nx, ny, nz));
    let mut curl_y = Array3::<T>::zeros((nx, ny, nz));
    let mut curl_z = Array3::<T>::zeros((nx, ny, nz));

    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();

    let dx_inv = T::one() / T::from(grid.dx).unwrap();
    let dy_inv = T::one() / T::from(grid.dy).unwrap();
    let dz_inv = T::one() / T::from(grid.dz).unwrap();

    // Compute curl in interior points
    for i in stencil_radius..nx - stencil_radius {
        for j in stencil_radius..ny - stencil_radius {
            for k in stencil_radius..nz - stencil_radius {
                let mut dvz_dy = T::zero();
                let mut dvy_dz = T::zero();
                let mut dvx_dz = T::zero();
                let mut dvz_dx = T::zero();
                let mut dvy_dx = T::zero();
                let mut dvx_dy = T::zero();

                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    
                    // For curl_x = ∂vz/∂y - ∂vy/∂z
                    dvz_dy = dvz_dy + coeff * (vz[[i, j + offset, k]] - vz[[i, j - offset, k]]);
                    dvy_dz = dvy_dz + coeff * (vy[[i, j, k + offset]] - vy[[i, j, k - offset]]);
                    
                    // For curl_y = ∂vx/∂z - ∂vz/∂x
                    dvx_dz = dvx_dz + coeff * (vx[[i, j, k + offset]] - vx[[i, j, k - offset]]);
                    dvz_dx = dvz_dx + coeff * (vz[[i + offset, j, k]] - vz[[i - offset, j, k]]);
                    
                    // For curl_z = ∂vy/∂x - ∂vx/∂y
                    dvy_dx = dvy_dx + coeff * (vy[[i + offset, j, k]] - vy[[i - offset, j, k]]);
                    dvx_dy = dvx_dy + coeff * (vx[[i, j + offset, k]] - vx[[i, j - offset, k]]);
                }

                curl_x[[i, j, k]] = dvz_dy * dy_inv - dvy_dz * dz_inv;
                curl_y[[i, j, k]] = dvx_dz * dz_inv - dvz_dx * dx_inv;
                curl_z[[i, j, k]] = dvy_dx * dx_inv - dvx_dy * dy_inv;
            }
        }
    }

    Ok((curl_x, curl_y, curl_z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_curl_constant_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let vx = Array3::<f64>::ones((5, 5, 5));
        let vy = Array3::<f64>::ones((5, 5, 5));
        let vz = Array3::<f64>::ones((5, 5, 5));

        let (curl_x, curl_y, curl_z) = curl(&vx.view(), &vy.view(), &vz.view(), &grid, SpatialOrder::Second)?;

        // Curl of constant field should be zero
        assert_relative_eq!(curl_x[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_y[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_z[[2, 2, 2]], 0.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_curl_simple_rotation() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let mut vx = Array3::<f64>::zeros((5, 5, 5));
        let mut vy = Array3::<f64>::zeros((5, 5, 5));
        let vz = Array3::<f64>::zeros((5, 5, 5));

        // Simple rotation field: vx = -y, vy = x, vz = 0
        // Curl should be (0, 0, 2)
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    vx[[i, j, k]] = -(j as f64);
                    vy[[i, j, k]] = i as f64;
                }
            }
        }

        let (curl_x, curl_y, curl_z) = curl(&vx.view(), &vy.view(), &vz.view(), &grid, SpatialOrder::Second)?;

        // Check interior point - should have curl_z = 2
        assert_relative_eq!(curl_x[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_y[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_z[[2, 2, 2]], 2.0, epsilon = 1e-10);

        Ok(())
    }
}