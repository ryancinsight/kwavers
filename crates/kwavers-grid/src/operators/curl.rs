//! Cartesian curl of a sampled vector field.
use super::coefficients::FdAccuracyOrder;
use super::stencil::vector::{self, Axis, VectorDerivative};
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

struct Curl;

impl VectorDerivative<3> for Curl {
    #[inline]
    fn evaluate<T: FloatElement>(partial: impl Fn(Axis, Axis) -> T) -> [T; 3] {
        use Axis::{X, Y, Z};
        [
            partial(Z, Y) - partial(Y, Z),
            partial(X, Z) - partial(Z, X),
            partial(Y, X) - partial(X, Y),
        ]
    }
}

/// Compute curl with centered finite differences and a zero output halo.
/// Curl returns the three components in x, y, z order.
/// For velocity in metres per second, the result is vorticity in inverse seconds.
///
/// # Errors
/// Returns a dimension mismatch if any input shape differs from the grid.
pub fn curl<T: FloatElement + Default>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)> {
    let [x, y, z] = vector::owned::<Curl, _, 3>([vx, vy, vz], grid, order)?;
    Ok((x, y, z))
}

/// Write [curl] into caller-owned row-major storage, with z varying fastest.
/// Every output cell is overwritten. The halo is zero on all six faces.
///
/// # Errors
/// Rejects an invalid input shape or output length before writing any results.
pub fn curl_into<T: FloatElement>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    output: [&mut [T]; 3],
) -> KwaversResult<()> {
    vector::into::<Curl, _, 3>([vx, vy, vz], grid, order, output)
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
