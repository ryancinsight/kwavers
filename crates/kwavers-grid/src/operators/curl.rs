//! Curl operations module

use super::coefficients::{FDCoefficients, FdAccuracyOrder};
use crate::compat::leto::{Array3, ArrayView3};
use crate::Grid;
use kwavers_core::error::KwaversResult;
use leto::Array3 as LetoArray3;
use eunomia::FloatElement;

/// Compute curl of a vector field
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
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
    let shape = vx.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Validate grid compatibility and vector field consistency
    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(kwavers_core::error::KwaversError::Grid(
            kwavers_core::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    if vy.shape() != shape || vz.shape() != shape {
        return Err(kwavers_core::error::KwaversError::Grid(
            kwavers_core::error::GridError::DimensionMismatch {
                expected: "Vector field components must have same dimensions".to_owned(),
                actual: format!(
                    "vx: {:?}, vy: {:?}, vz: {:?}",
                    vx.shape(),
                    vy.shape(),
                    vz.shape()
                ),
            },
        ));
    }

    let mut curl_x = Array3::<T>::zeros([nx, ny, nz]);
    let mut curl_y = Array3::<T>::zeros([nx, ny, nz]);
    let mut curl_z = Array3::<T>::zeros([nx, ny, nz]);

    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();

    let dx_inv = T::from_f64(1.0) / T::from_f64(grid.dx as f64);
    let dy_inv = T::from_f64(1.0) / T::from_f64(grid.dy as f64);
    let dz_inv = T::from_f64(1.0) / T::from_f64(grid.dz as f64);

    // Compute curl in interior points
    for i in stencil_radius..nx - stencil_radius {
        for j in stencil_radius..ny - stencil_radius {
            for k in stencil_radius..nz - stencil_radius {
                let mut dvz_dy = T::from_f64(0.0);
                let mut dvy_dz = T::from_f64(0.0);
                let mut dvx_dz = T::from_f64(0.0);
                let mut dvz_dx = T::from_f64(0.0);
                let mut dvy_dx = T::from_f64(0.0);
                let mut dvx_dy = T::from_f64(0.0);

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

/// Compute curl of a leto 3D vector field.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub fn curl_leto<T>(
    vx: &LetoArray3<T>,
    vy: &LetoArray3<T>,
    vz: &LetoArray3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<(LetoArray3<T>, LetoArray3<T>, LetoArray3<T>)>
where
    T: FloatElement + Clone + Send + Sync + Default,
{
    let vx_view = vx.view();
    let vy_view = vy.view();
    let vz_view = vz.view();
    curl(&vx_view, &vy_view, &vz_view, grid, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
