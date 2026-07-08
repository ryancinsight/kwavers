//! Divergence operations module

use super::coefficients::{FDCoefficients, FdAccuracyOrder};
use crate::compat::ndarray::{Array3, ArrayView3};
use crate::Grid;
use kwavers_core::error::KwaversResult;
use leto::Array3 as LetoArray3;
use num_traits::Float;

/// Compute divergence of a vector field
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
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
    T: Float + Clone + Send + Sync + Default,
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

    let mut divergence = Array3::<T>::zeros([nx, ny, nz]);
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

/// Compute divergence of a leto 3D vector field.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub fn divergence_leto<T>(
    vx: &LetoArray3<T>,
    vy: &LetoArray3<T>,
    vz: &LetoArray3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<LetoArray3<T>>
where
    T: Float + Clone + Send + Sync + Default,
{
    let vx_view = vx.view();
    let vy_view = vy.view();
    let vz_view = vz.view();
    divergence(&vx_view, &vy_view, &vz_view, grid, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
