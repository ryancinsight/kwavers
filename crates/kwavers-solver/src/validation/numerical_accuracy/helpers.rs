use leto::{
    Array1,
    Array3,
};

pub(crate) fn compute_laplacian_1d(field: &Array1<f64>, dx: f64) -> Array1<f64> {
    let n = (field.shape()[0] * field.shape()[1] * field.shape()[2]);
    let mut laplacian = Array1::zeros(n);
    let dx2_inv = 1.0 / (dx * dx);

    for i in 1..n - 1 {
        laplacian[i] = (field[i + 1] - 2.0 * field[i] + field[i - 1]) * dx2_inv;
    }

    laplacian[0] = (field[1] - field[0]) * dx2_inv;
    laplacian[n - 1] = (field[n - 2] - field[n - 1]) * dx2_inv;

    laplacian
}

/// Compute the 3D Laplacian on a periodic domain using the 7-point
/// second-order central difference stencil.
///
/// For each interior and boundary point (periodic wrapping):
/// ```text
///   ∇²u[i,j,k] = (u[i+1,j,k] + u[i-1,j,k]
///                + u[i,j+1,k] + u[i,j-1,k]
///                + u[i,j,k+1] + u[i,j,k-1]
///                − 6·u[i,j,k]) / Δx²
/// ```
/// Accuracy: O(Δx²)  (Fornberg 1988, Eq. 3.1).
///
/// With periodic BCs, the total sum Σ∇²u = 0 (discrete divergence theorem),
/// so any conserved-sum PDE (e.g., heat equation ∂T/∂t = α∇²T with Σu = const)
/// is preserved to floating-point precision.
pub(crate) fn compute_laplacian_3d(field: &Array3<f64>, dx: f64) -> Array3<f64> {
    let [nx, ny, nz] = field.shape();
    let mut lap = Array3::zeros((nx, ny, nz));
    let dx2_inv = 1.0 / (dx * dx);
    for i in 0..nx {
        let ip1 = (i + 1) % nx;
        let im1 = (i + nx - 1) % nx;
        for j in 0..ny {
            let jp1 = (j + 1) % ny;
            let jm1 = (j + ny - 1) % ny;
            for k in 0..nz {
                let kp1 = (k + 1) % nz;
                let km1 = (k + nz - 1) % nz;
                lap[[i, j, k]] = (field[[ip1, j, k]]
                    + field[[im1, j, k]]
                    + field[[i, jp1, k]]
                    + field[[i, jm1, k]]
                    + field[[i, j, kp1]]
                    + field[[i, j, km1]]
                    - 6.0 * field[[i, j, k]])
                    * dx2_inv;
            }
        }
    }
    lap
}
