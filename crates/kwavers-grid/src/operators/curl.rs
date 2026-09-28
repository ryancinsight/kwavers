//! Curl operations module

use super::coefficients::{FDCoefficients, FdAccuracyOrder};
use super::stencil::{centered_first_derivative_sum, validate_vector_field_shapes};
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

/// Compute curl of a vector field
/// # Errors
/// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub fn curl<T>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: FloatElement + Clone + Send + Sync + Default,
{
    let [nx, ny, nz] = validate_vector_field_shapes(vx, vy, vz, grid)?;

    let mut curl_x = Array3::<T>::zeros([nx, ny, nz]);
    let mut curl_y = Array3::<T>::zeros([nx, ny, nz]);
    let mut curl_z = Array3::<T>::zeros([nx, ny, nz]);

    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();

    let dx_inv = T::from_f64(1.0) / T::from_f64(grid.dx);
    let dy_inv = T::from_f64(1.0) / T::from_f64(grid.dy);
    let dz_inv = T::from_f64(1.0) / T::from_f64(grid.dz);

    // Compute curl in interior points
    for i in stencil_radius..nx - stencil_radius {
        for j in stencil_radius..ny - stencil_radius {
            for k in stencil_radius..nz - stencil_radius {
                let dvz_dy = centered_first_derivative_sum(&coeffs, |offset| {
                    vz[[i, j + offset, k]] - vz[[i, j - offset, k]]
                });
                let dvy_dz = centered_first_derivative_sum(&coeffs, |offset| {
                    vy[[i, j, k + offset]] - vy[[i, j, k - offset]]
                });
                let dvx_dz = centered_first_derivative_sum(&coeffs, |offset| {
                    vx[[i, j, k + offset]] - vx[[i, j, k - offset]]
                });
                let dvz_dx = centered_first_derivative_sum(&coeffs, |offset| {
                    vz[[i + offset, j, k]] - vz[[i - offset, j, k]]
                });
                let dvy_dx = centered_first_derivative_sum(&coeffs, |offset| {
                    vy[[i + offset, j, k]] - vy[[i - offset, j, k]]
                });
                let dvx_dy = centered_first_derivative_sum(&coeffs, |offset| {
                    vx[[i, j + offset, k]] - vx[[i, j - offset, k]]
                });

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
    use eunomia::assert_relative_eq;

    #[test]
    fn test_curl_constant_field() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let vx = Array3::<f64>::ones([5, 5, 5]);
        let vy = Array3::<f64>::ones([5, 5, 5]);
        let vz = Array3::<f64>::ones([5, 5, 5]);

        let (curl_x, curl_y, curl_z) = curl(
            &vx.view(),
            &vy.view(),
            &vz.view(),
            &grid,
            FdAccuracyOrder::Second,
        )?;

        // Curl of constant field should be zero
        assert_relative_eq!(curl_x[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_y[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_z[[2, 2, 2]], 0.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_curl_simple_rotation() -> KwaversResult<()> {
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0)?;
        let mut vx = Array3::<f64>::zeros([5, 5, 5]);
        let mut vy = Array3::<f64>::zeros([5, 5, 5]);
        let vz = Array3::<f64>::zeros([5, 5, 5]);

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

        let (curl_x, curl_y, curl_z) = curl(
            &vx.view(),
            &vy.view(),
            &vz.view(),
            &grid,
            FdAccuracyOrder::Second,
        )?;

        // Check interior point - should have curl_z = 2
        assert_relative_eq!(curl_x[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_y[[2, 2, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(curl_z[[2, 2, 2]], 2.0, epsilon = 1e-10);

        Ok(())
    }
}
