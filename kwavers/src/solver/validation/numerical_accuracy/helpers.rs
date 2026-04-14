use ndarray::{Array1, Array3};
use std::f64::consts::PI;

pub(crate) fn compute_laplacian_1d(field: &Array1<f64>, dx: f64) -> Array1<f64> {
    let n = field.len();
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
    let (nx, ny, nz) = field.dim();
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

#[allow(dead_code)]
/// Helper function to compute phase error using least squares fit
pub(crate) fn compute_phase_error_lsq(
    pressure: &Array3<f64>,
    k: f64,
    dx: f64,
    n: usize,
    axis: usize, // 0=x, 1=y, 2=z
) -> f64 {
    // Extract line profile along specified axis
    let mut sum_y_sin = 0.0;
    let mut sum_y_cos = 0.0;
    let mut sum_sin2 = 0.0;
    let mut sum_cos2 = 0.0;
    let mut sum_sin_cos = 0.0;

    // Use middle slice for analysis
    let mid = n / 2;

    for i in n / 4..3 * n / 4 {
        let x = i as f64 * dx;
        let val = match axis {
            0 => pressure[[i, mid, 0]],   // x-axis
            1 => pressure[[mid, i, 0]],   // y-axis
            _ => pressure[[mid, mid, i]], // z-axis
        };
        let s = (k * x).sin();
        let c = (k * x).cos();

        sum_y_sin += val * s;
        sum_y_cos += val * c;
        sum_sin2 += s * s;
        sum_cos2 += c * c;
        sum_sin_cos += s * c;
    }

    let det = sum_sin2 * sum_cos2 - sum_sin_cos * sum_sin_cos;
    if det.abs() < 1e-10 {
        return PI; // Return max error if singular
    }
    let a = (sum_y_sin * sum_cos2 - sum_y_cos * sum_sin_cos) / det;
    let b = (sum_y_cos * sum_sin2 - sum_y_sin * sum_sin_cos) / det;

    // Phase error = atan2(b, a)
    b.atan2(a).abs()
}
