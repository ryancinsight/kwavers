//! Boundary extrapolation for elastography reconstruction arrays.
//!
//! Reference: Numerical Recipes (2007), *Boundary Conditions*, Ch. 19.

use ndarray::Array3;

/// Fill boundary voxels with nearest interior values.
///
/// Propagates interior values to all six domain faces using nearest-neighbour
/// (zero-order) extrapolation. Edges and corners inherit from the filled faces.
pub fn fill_boundaries(array: &mut Array3<f64>) {
    let (nx, ny, nz) = array.dim();

    // Fill i=0 and i=nx-1 faces.
    for k in 0..nz {
        for j in 0..ny {
            array[[0, j, k]] = array[[1, j, k]];
            array[[nx - 1, j, k]] = array[[nx - 2, j, k]];
        }
    }

    // Fill j=0 and j=ny-1 faces.
    for k in 0..nz {
        for i in 0..nx {
            array[[i, 0, k]] = array[[i, 1, k]];
            array[[i, ny - 1, k]] = array[[i, ny - 2, k]];
        }
    }

    // Fill k=0 and k=nz-1 faces.
    for j in 0..ny {
        for i in 0..nx {
            array[[i, j, 0]] = array[[i, j, 1]];
            array[[i, j, nz - 1]] = array[[i, j, nz - 2]];
        }
    }
}
