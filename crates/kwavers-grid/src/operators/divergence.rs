//! Divergence operations module

use super::coefficients::{FDCoefficients, FdAccuracyOrder};
use super::stencil::{centered_first_derivative_sum, validate_vector_field_shapes};
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

/// Compute divergence of a vector field
/// # Errors
/// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub fn divergence<T>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<Array3<T>>
where
    T: FloatElement + Clone + Send + Sync + Default,
{
    let [nx, ny, nz] = validate_vector_field_shapes(vx, vy, vz, grid)?;

    let mut divergence = Array3::<T>::zeros([nx, ny, nz]);
    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();

    let dx_inv = T::from_f64(1.0) / T::from_f64(grid.dx);
    let dy_inv = T::from_f64(1.0) / T::from_f64(grid.dy);
    let dz_inv = T::from_f64(1.0) / T::from_f64(grid.dz);

    // Compute divergence in interior points
    for i in stencil_radius..nx - stencil_radius {
        for j in stencil_radius..ny - stencil_radius {
            for k in stencil_radius..nz - stencil_radius {
                let div_x = centered_first_derivative_sum(&coeffs, |offset| {
                    vx[[i + offset, j, k]] - vx[[i - offset, j, k]]
                });
                let div_y = centered_first_derivative_sum(&coeffs, |offset| {
                    vy[[i, j + offset, k]] - vy[[i, j - offset, k]]
                });
                let div_z = centered_first_derivative_sum(&coeffs, |offset| {
                    vz[[i, j, k + offset]] - vz[[i, j, k - offset]]
                });

                divergence[[i, j, k]] = div_x * dx_inv + div_y * dy_inv + div_z * dz_inv;
            }
        }
    }

    Ok(divergence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_relative_eq;

    #[test]
    fn test_divergence_constant_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let vx = Array3::<f64>::ones([5, 5, 5]);
        let vy = Array3::<f64>::ones([5, 5, 5]);
        let vz = Array3::<f64>::ones([5, 5, 5]);

        let div = divergence(
            &vx.view(),
            &vy.view(),
            &vz.view(),
            &grid,
            FdAccuracyOrder::Second,
        )?;

        // Divergence of constant field should be zero in interior
        assert_relative_eq!(div[[2, 2, 2]], 0.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_divergence_linear_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let mut vx = Array3::<f64>::zeros([5, 5, 5]);
        let mut vy = Array3::<f64>::zeros([5, 5, 5]);
        let mut vz = Array3::<f64>::zeros([5, 5, 5]);

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
            FdAccuracyOrder::Second,
        )?;

        // Check interior point
        assert_relative_eq!(div[[2, 2, 2]], 6.0, epsilon = 1e-10);

        Ok(())
    }
}
