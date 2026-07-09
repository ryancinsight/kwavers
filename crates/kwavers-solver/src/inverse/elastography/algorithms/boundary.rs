//! Boundary extrapolation for elastography reconstruction arrays.
//!
//! Reference: Numerical Recipes (2007), *Boundary Conditions*, Ch. 19.

use super::volume::Volume3;

/// Fill boundary voxels with nearest interior values.
///
/// Propagates interior values to all six domain faces using nearest-neighbour
/// (zero-order) extrapolation. Edges and corners inherit from the filled faces.
///
/// A singleton axis (`n == 1`, e.g. a 2-D plane-strain `nz = 1` field) has no
/// interior to extrapolate from along that axis, so its face fill is skipped —
/// the single layer is already its own boundary. This keeps the function panic-
/// free on 2-D inputs.
pub fn fill_boundaries<V>(array: &mut V)
where
    V: Volume3,
{
    let (nx, ny, nz) = array.dimensions();

    // Fill i=0 and i=nx-1 faces.
    if nx >= 2 {
        for k in 0..nz {
            for j in 0..ny {
                array.set_value([0, j, k], array.value([1, j, k]));
                array.set_value([nx - 1, j, k], array.value([nx - 2, j, k]));
            }
        }
    }

    // Fill j=0 and j=ny-1 faces.
    if ny >= 2 {
        for k in 0..nz {
            for i in 0..nx {
                array.set_value([i, 0, k], array.value([i, 1, k]));
                array.set_value([i, ny - 1, k], array.value([i, ny - 2, k]));
            }
        }
    }

    // Fill k=0 and k=nz-1 faces.
    if nz >= 2 {
        for j in 0..ny {
            for i in 0..nx {
                array.set_value([i, j, 0], array.value([i, j, 1]));
                array.set_value([i, j, nz - 1], array.value([i, j, nz - 2]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array3;

    #[test]
    fn fill_boundaries_handles_singleton_z_plane() {
        // 2-D plane-strain field (nz = 1) must not panic; the single z-layer is
        // its own boundary, and the in-plane faces still extrapolate.
        let mut a = Array3::<f64>::zeros((4, 4, 1));
        for i in 0..4 {
            for j in 0..4 {
                a[[i, j, 0]] = (i * 10 + j) as f64;
            }
        }
        fill_boundaries(&mut a);
        // In-plane faces inherited from the nearest interior column/row.
        assert_eq!(a[[0, 1, 0]], a[[1, 1, 0]]);
        assert_eq!(a[[3, 1, 0]], a[[2, 1, 0]]);
        assert_eq!(a[[1, 0, 0]], a[[1, 1, 0]]);
        assert_eq!(a[[1, 3, 0]], a[[1, 2, 0]]);
    }
}
