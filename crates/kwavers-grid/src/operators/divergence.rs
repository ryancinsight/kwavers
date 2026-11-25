//! Cartesian divergence of a sampled vector field.
use super::coefficients::FdAccuracyOrder;
use super::stencil::vector::{self, Axis, VectorDerivative};
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

struct Divergence;

impl VectorDerivative<1> for Divergence {
    #[inline]
    fn evaluate<T: FloatElement>(partial: impl Fn(Axis, Axis) -> T) -> [T; 1] {
        use Axis::{X, Y, Z};
        [partial(X, X) + partial(Y, Y) + partial(Z, Z)]
    }
}

/// Compute divergence with centered finite differences and a zero output halo.
/// Divergence sums the three diagonal spatial derivatives.
/// For velocity in metres per second, the result is volumetric strain rate in inverse seconds.
///
/// # Errors
/// Returns a dimension mismatch if any input shape differs from the grid.
pub fn divergence<T: FloatElement + Default>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<Array3<T>> {
    let [result] = vector::owned::<Divergence, _, 1>([vx, vy, vz], grid, order)?;
    Ok(result)
}

/// Write [divergence] into caller-owned row-major storage, with z varying fastest.
/// Every output cell is overwritten. The halo is zero on all six faces.
///
/// # Errors
/// Rejects an invalid input shape or output length before writing any results.
pub fn divergence_into<T: FloatElement>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    output: &mut [T],
) -> KwaversResult<()> {
    vector::into::<Divergence, _, 1>([vx, vy, vz], grid, order, [output])
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
