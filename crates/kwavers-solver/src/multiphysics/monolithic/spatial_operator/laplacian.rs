use leto::{
    Array3,
};

/// Compute the 3-D Laplacian ∇²f using second-order central differences.
///
/// ## Algorithm
///
/// ```text
/// ∇²f[i,j,k] ≈ (f[i+1,j,k] - 2f[i,j,k] + f[i-1,j,k]) / dx²
///             + (f[i,j+1,k] - 2f[i,j,k] + f[i,j-1,k]) / dy²
///             + (f[i,j,k+1] - 2f[i,j,k] + f[i,j,k-1]) / dz²
/// ```
///
/// Truncation error: `O(dx², dy², dz²)`. Boundary nodes use homogeneous
/// Neumann ghost-cell conditions.
///
/// The caller owns `lap`, and the kernel overwrites every output cell exactly
/// once. The input is generic over `ndarray` storage, so owned arrays and
/// borrowed block views monomorphize to direct indexing code.
///
/// Reference: LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary
/// and Partial Differential Equations*. SIAM. §1.3.
pub(in crate::multiphysics::monolithic) fn laplacian_3d_into<S>(
    field: &ArrayBase<S, Ix3>,
    grid_dims: (usize, usize, usize),
    dx: f64,
    dy: f64,
    dz: f64,
    lap: &mut Array3<f64>,
) where
    S: Data<Elem = f64>,
{
    let (nx, ny, nz) = field.dim();
    assert_eq!(
        lap.dim(),
        (nx, ny, nz),
        "Laplacian output must match input field dimensions"
    );
    let _ = grid_dims;

    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    let inv_dz2 = 1.0 / (dz * dz);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let d2x = if nx > 2 {
                    let im = if i == 0 { 0 } else { i - 1 };
                    let ip = if i == nx - 1 { nx - 1 } else { i + 1 };
                    (2.0f64.mul_add(-field[[i, j, k]], field[[ip, j, k]]) + field[[im, j, k]])
                        * inv_dx2
                } else {
                    0.0
                };
                let d2y = if ny > 2 {
                    let jm = if j == 0 { 0 } else { j - 1 };
                    let jp = if j == ny - 1 { ny - 1 } else { j + 1 };
                    (2.0f64.mul_add(-field[[i, j, k]], field[[i, jp, k]]) + field[[i, jm, k]])
                        * inv_dy2
                } else {
                    0.0
                };
                let d2z = if nz > 2 {
                    let km = if k == 0 { 0 } else { k - 1 };
                    let kp = if k == nz - 1 { nz - 1 } else { k + 1 };
                    (2.0f64.mul_add(-field[[i, j, k]], field[[i, j, kp]]) + field[[i, j, km]])
                        * inv_dz2
                } else {
                    0.0
                };

                lap[[i, j, k]] = d2x + d2y + d2z;
            }
        }
    }
}
