//! Divergence operations module
//!
//! This module owns divergence computation knowledge following GRASP principles.
//! Single responsibility: Computing vector field divergence.

use super::coefficients::{FDCoefficients, SpatialOrder};
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, ArrayView3};
use num_traits::Float;

/// Compute divergence of a vector field
///
/// Returns ∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
///
/// # Arguments
/// * `vx`, `vy`, `vz` - Components of the vector field
/// * `grid` - Grid information for spatial discretization
/// * `order` - Finite difference accuracy order
///
/// # Generic Parameters
/// * `T` - Float type (f32, f64) implementing num_traits::Float
///
/// # Errors
/// Returns error if field dimensions don't match grid.
pub fn divergence<T>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<Array3<T>>
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
                expected: "Vector field components must have same dimensions".to_string(),
                actual: format!(
                    "vx: {:?}, vy: {:?}, vz: {:?}",
                    vx.shape(),
                    vy.shape(),
                    vz.shape()
                ),
            },
        ));
    }

    let mut divergence = Array3::<T>::zeros((nx, ny, nz));
    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();

    let dx_inv = T::one() / T::from(grid.dx).unwrap();
    let dy_inv = T::one() / T::from(grid.dy).unwrap();
    let dz_inv = T::one() / T::from(grid.dz).unwrap();

    // Compute divergence in interior points
    for i in stencil_radius..nx - stencil_radius {
        for j in stencil_radius..ny - stencil_radius {
            for k in stencil_radius..nz - stencil_radius {
                let mut div_x = T::zero();
                let mut div_y = T::zero();
                let mut div_z = T::zero();

                // ∂vx/∂x
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    div_x = div_x + coeff * (vx[[i + offset, j, k]] - vx[[i - offset, j, k]]);
                }

                // ∂vy/∂y
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    div_y = div_y + coeff * (vy[[i, j + offset, k]] - vy[[i, j - offset, k]]);
                }

                // ∂vz/∂z
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    div_z = div_z + coeff * (vz[[i, j, k + offset]] - vz[[i, j, k - offset]]);
                }

                divergence[[i, j, k]] = div_x * dx_inv + div_y * dy_inv + div_z * dz_inv;
            }
        }
    }

    Ok(divergence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_divergence_constant_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let vx = Array3::<f64>::ones((5, 5, 5));
        let vy = Array3::<f64>::ones((5, 5, 5));
        let vz = Array3::<f64>::ones((5, 5, 5));

        let div = divergence(
            &vx.view(),
            &vy.view(),
            &vz.view(),
            &grid,
            SpatialOrder::Second,
        )?;

        // Divergence of constant field should be zero in interior
        assert_relative_eq!(div[[2, 2, 2]], 0.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_divergence_linear_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let mut vx = Array3::<f64>::zeros((5, 5, 5));
        let mut vy = Array3::<f64>::zeros((5, 5, 5));
        let mut vz = Array3::<f64>::zeros((5, 5, 5));

        // Create linear field: vx = x, vy = 2y, vz = 3z
        // Divergence should be 1 + 2 + 3 = 6
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    vx[[i, j, k]] = i as f64;
                    vy[[i, j, k]] = 2.0 * j as f64;
                    vz[[i, j, k]] = 3.0 * k as f64;
                }
            }
        }

        let div = divergence(
            &vx.view(),
            &vy.view(),
            &vz.view(),
            &grid,
            SpatialOrder::Second,
        )?;

        // Check interior point
        assert_relative_eq!(div[[2, 2, 2]], 6.0, epsilon = 1e-10);

        Ok(())
    }
}
