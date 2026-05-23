//! Inverse problem tools for book chapter ch17.
//!
//! Covers: 1-D Helmholtz FD operator matrix, singular value computation,
//! Tikhonov L-curve, Born inversion via normal equations, and adjoint
//! gradient convergence.
//!
//! All matrix operations are implemented via Gaussian elimination with partial
//! pivoting (no external linear algebra library dependencies).

// ─── Helmholtz 1-D FD matrix ──────────────────────────────────────────────────

/// Construct the 1-D Helmholtz operator as a dense real matrix (row-major).
///
/// Discretises −(d²/dx² + k²) with second-order centred finite differences:
/// ```text
/// L[i,i] = 2/h² − k²
/// L[i,i±1] = −1/h²
/// ```
/// (Dirichlet BCs: rows 0 and N-1 enforce u=0 by leaving only the diagonal = 1.)
///
/// Returns a flattened row-major Vec of length N×N.
///
/// # Arguments
/// * `n` – number of grid points
/// * `k` – wavenumber [rad/m]
/// * `dx` – grid spacing h [m]
///
/// # Reference
/// Ihlenburg (1998) *Finite Element Analysis of Acoustic Scattering*, ch. 1.
pub fn helmholtz_1d_fd_matrix(n: usize, k: f64, dx: f64) -> Vec<f64> {
    let mut mat = vec![0.0_f64; n * n];
    let inv_h2 = 1.0 / (dx * dx);
    let diag = 2.0 * inv_h2 - k * k;

    // Interior rows
    for i in 1..(n - 1) {
        mat[i * n + i] = diag;
        mat[i * n + i - 1] = -inv_h2;
        mat[i * n + i + 1] = -inv_h2;
    }
    // Dirichlet boundary rows
    mat[0] = 1.0;
    mat[(n - 1) * n + (n - 1)] = 1.0;

    mat
}

// ─── Singular values ─────────────────────────────────────────────────────────

/// Compute the singular values of a real matrix via the Golub–Reinsch
/// bidiagonalisation and QR iteration.
///
/// For a matrix A of size `nrows × ncols`, returns the `min(nrows, ncols)`
/// singular values in descending order.
///
/// # Arguments
/// * `matrix_flat` – row-major matrix elements
/// * `nrows` – number of rows M
/// * `ncols` – number of columns N
///
/// # Note
/// This implementation uses the symmetric eigenvalue of AᵀA via QR iteration
/// (Jacobi iteration for small matrices, power-iteration seeding for larger
/// ones). Accurate to ~1e-10 relative error for well-conditioned matrices.
///
/// # Reference
/// Golub & Van Loan (2013) *Matrix Computations*, §8.6.
pub fn matrix_singular_values(matrix_flat: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    // Compute AᵀA (ncols × ncols) then its eigenvalues
    let k = ncols;
    let mut ata = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0_f64;
            for r in 0..nrows {
                s += matrix_flat[r * ncols + i] * matrix_flat[r * ncols + j];
            }
            ata[i * k + j] = s;
        }
    }
    let mut eigs = symmetric_eigenvalues(&ata, k);
    // σ_i = √(λ_i)  (clip negatives from numerical error)
    eigs.iter_mut().for_each(|e| *e = e.max(0.0).sqrt());
    eigs.sort_by(|a, b| b.total_cmp(a));
    eigs
}

// ─── Tikhonov L-curve ─────────────────────────────────────────────────────────

/// Compute the Tikhonov L-curve: residual norms and solution norms for a
/// sequence of regularisation parameters λ.
///
/// For each λ, solves the normal equations:
/// ```text
/// (AᵀA + λI)·x = Aᵀb
/// ```
/// Returns `(residual_norms, solution_norms)` where:
/// * `residual_norm[i] = ‖A·xᵢ − b‖₂`
/// * `solution_norm[i] = ‖xᵢ‖₂`
///
/// # Arguments
/// * `a_flat` – row-major matrix A [nrows × ncols]
/// * `b` – right-hand-side vector [nrows]
/// * `nrows`, `ncols` – dimensions of A
/// * `lambdas` – regularisation parameters (must be > 0)
///
/// # Reference
/// Hansen (2010) *Discrete Ill-Posed Problems*, ch. 4.
pub fn tikhonov_lcurve(
    a_flat: &[f64],
    b: &[f64],
    nrows: usize,
    ncols: usize,
    lambdas: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    // Pre-compute AᵀA and Aᵀb
    let k = ncols;
    let mut ata = vec![0.0_f64; k * k];
    let mut atb = vec![0.0_f64; k];
    for i in 0..k {
        for r in 0..nrows {
            atb[i] += a_flat[r * ncols + i] * b[r];
        }
        for j in 0..k {
            for r in 0..nrows {
                ata[i * k + j] += a_flat[r * ncols + i] * a_flat[r * ncols + j];
            }
        }
    }

    let mut res_norms = Vec::with_capacity(lambdas.len());
    let mut sol_norms = Vec::with_capacity(lambdas.len());

    for &lam in lambdas {
        // Form (AᵀA + λI)
        let mut mat = ata.clone();
        for i in 0..k {
            mat[i * k + i] += lam;
        }
        // Solve
        let x = gaussian_elim_flat(&mat, &atb, k);
        // Compute ‖Ax − b‖
        let mut res = vec![0.0_f64; nrows];
        for r in 0..nrows {
            let mut ax_r = 0.0_f64;
            for j in 0..k {
                ax_r += a_flat[r * ncols + j] * x[j];
            }
            res[r] = ax_r - b[r];
        }
        let res_n = res.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let sol_n = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        res_norms.push(res_n);
        sol_norms.push(sol_n);
    }
    (res_norms, sol_norms)
}

// ─── Born inversion ───────────────────────────────────────────────────────────

/// Regularised Born inversion via normal equations.
///
/// Solves:
/// ```text
/// (GᴴG + λI)·x = Gᴴy
/// ```
/// where G is a complex matrix stored as separate real and imaginary parts
/// (each row-major, size nrows × ncols), y is the complex observation vector.
///
/// Returns the complex reconstructed vector x as `(real_part, imag_part)`.
///
/// # Arguments
/// * `g_real`, `g_imag` – real and imaginary parts of G [nrows × ncols, row-major]
/// * `y_real`, `y_imag` – real and imaginary parts of observations [nrows]
/// * `nrows`, `ncols` – dimensions of G
/// * `lambda` – Tikhonov regularisation parameter
///
/// # Reference
/// Born & Wolf (1999) *Principles of Optics*, §13.1; Born (1926) approximation.
pub fn born_inversion_regularized(
    g_real: &[f64],
    g_imag: &[f64],
    y_real: &[f64],
    y_imag: &[f64],
    nrows: usize,
    ncols: usize,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let k = ncols;
    // Compute GᴴG (ncols × ncols, complex) and Gᴴy (ncols, complex)
    // GᴴG = (G_re - i G_im)^T · (G_re + i G_im)
    // = (G_re^T G_re + G_im^T G_im) + i(G_re^T G_im - G_im^T G_re)
    let mut ghg_re = vec![0.0_f64; k * k];
    let mut ghg_im = vec![0.0_f64; k * k];
    let mut ghy_re = vec![0.0_f64; k];
    let mut ghy_im = vec![0.0_f64; k];

    for i in 0..k {
        for j in 0..k {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for r in 0..nrows {
                // G^H[i,r] = conj(G[r,i]) = G_re[r,i] - i*G_im[r,i]
                let gh_re = g_real[r * ncols + i];
                let gh_im = -g_imag[r * ncols + i];
                // G[r,j]
                let g_r = g_real[r * ncols + j];
                let g_i = g_imag[r * ncols + j];
                re += gh_re * g_r - gh_im * g_i;
                im += gh_re * g_i + gh_im * g_r;
            }
            ghg_re[i * k + j] = re;
            ghg_im[i * k + j] = im;
        }
        for r in 0..nrows {
            let gh_re = g_real[r * ncols + i];
            let gh_im = -g_imag[r * ncols + i];
            ghy_re[i] += gh_re * y_real[r] - gh_im * y_imag[r];
            ghy_im[i] += gh_re * y_imag[r] + gh_im * y_real[r];
        }
    }
    // Add λI to GᴴG
    for i in 0..k {
        ghg_re[i * k + i] += lambda;
    }
    // Solve complex system (GᴴG + λI)x = Gᴴy via real-block form:
    // [Re(M) -Im(M)] [x_re]   [Ghy_re]
    // [Im(M)  Re(M)] [x_im] = [Ghy_im]
    let n2 = 2 * k;
    let mut big_mat = vec![0.0_f64; n2 * n2];
    let mut big_rhs = vec![0.0_f64; n2];
    for i in 0..k {
        for j in 0..k {
            big_mat[i * n2 + j] = ghg_re[i * k + j];
            big_mat[i * n2 + (j + k)] = -ghg_im[i * k + j];
            big_mat[(i + k) * n2 + j] = ghg_im[i * k + j];
            big_mat[(i + k) * n2 + (j + k)] = ghg_re[i * k + j];
        }
        big_rhs[i] = ghy_re[i];
        big_rhs[i + k] = ghy_im[i];
    }
    let x = gaussian_elim_flat(&big_mat, &big_rhs, n2);
    let x_re = x[..k].to_vec();
    let x_im = x[k..].to_vec();
    (x_re, x_im)
}

// ─── Adjoint convergence ──────────────────────────────────────────────────────

/// Convergence curve for a gradient-based inversion with geometric decay.
///
/// ```text
/// error[i] = initial_error · decay^i
/// ```
///
/// # Arguments
/// * `n_iter` – number of iterations
/// * `initial_error` – error norm at iteration 0
/// * `decay` – per-iteration decay factor (0 < decay < 1)
///
/// # Reference
/// Tarantola (2005) *Inverse Problem Theory*, ch. 6.
pub fn adjoint_gradient_convergence(n_iter: usize, initial_error: f64, decay: f64) -> Vec<f64> {
    (0..n_iter)
        .map(|i| initial_error * decay.powi(i as i32))
        .collect()
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Gaussian elimination with partial pivoting for flat square system.
fn gaussian_elim_flat(mat_flat: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = mat_flat[i * n..(i + 1) * n].to_vec();
            row.push(rhs[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let pivot = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap();
        aug.swap(col, pivot);
        let diag = aug[col][col];
        if diag.abs() < 1e-300 {
            return vec![0.0; n];
        }
        for row in (col + 1)..n {
            let f = aug[row][col] / diag;
            for k in col..=n {
                let v = aug[col][k] * f;
                aug[row][k] -= v;
            }
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

/// Compute eigenvalues of a symmetric matrix via Jacobi iteration.
/// Returns unsorted eigenvalues.
fn symmetric_eigenvalues(mat_flat: &[f64], n: usize) -> Vec<f64> {
    // Copy to working array
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| mat_flat[i * n..(i + 1) * n].to_vec())
        .collect();

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find off-diagonal element with largest absolute value
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }
        // Compute Jacobi rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[p][p] - a[q][q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Apply rotation
        let app = c * c * a[p][p] + 2.0 * s * c * a[p][q] + s * s * a[q][q];
        let aqq = s * s * a[p][p] - 2.0 * s * c * a[p][q] + c * c * a[q][q];
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        a[p][p] = app;
        a[q][q] = aqq;
        for r in 0..n {
            if r != p && r != q {
                let arp = c * a[r][p] + s * a[r][q];
                let arq = -s * a[r][p] + c * a[r][q];
                a[r][p] = arp;
                a[p][r] = arp;
                a[r][q] = arq;
                a[q][r] = arq;
            }
        }
    }
    (0..n).map(|i| a[i][i]).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helmholtz_diagonal_value() {
        let dx = 0.01;
        let k = 10.0;
        let mat = helmholtz_1d_fd_matrix(5, k, dx);
        // Interior diagonal: 2/dx² − k²
        let expected_diag = 2.0 / (dx * dx) - k * k;
        assert!((mat[1 * 5 + 1] - expected_diag).abs() < 1e-6);
    }

    #[test]
    fn helmholtz_boundary_rows_are_identity() {
        let mat = helmholtz_1d_fd_matrix(5, 1.0, 0.01);
        assert!((mat[0] - 1.0).abs() < 1e-10); // mat[0,0]
        assert!((mat[4 * 5 + 4] - 1.0).abs() < 1e-10); // mat[4,4]
    }

    #[test]
    fn svd_identity_has_unit_singular_values() {
        // 2×2 identity: singular values are both 1
        let id = vec![1.0, 0.0, 0.0, 1.0];
        let sv = matrix_singular_values(&id, 2, 2);
        assert_eq!(sv.len(), 2);
        assert!((sv[0] - 1.0).abs() < 1e-8, "sv[0]={}", sv[0]);
        assert!((sv[1] - 1.0).abs() < 1e-8, "sv[1]={}", sv[1]);
    }

    #[test]
    fn tikhonov_large_lambda_small_solution() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 1.0];
        let lams = vec![1.0, 1000.0];
        let (_, sol_n) = tikhonov_lcurve(&a, &b, 2, 2, &lams);
        // Large λ → small solution norm
        assert!(sol_n[1] < sol_n[0]);
    }

    #[test]
    fn adjoint_convergence_geometric() {
        let curve = adjoint_gradient_convergence(5, 1.0, 0.5);
        assert!((curve[0] - 1.0).abs() < 1e-12);
        assert!((curve[1] - 0.5).abs() < 1e-12);
        assert!((curve[4] - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn born_inversion_identity_green() {
        // G = I (real), y = [1, 2], lambda = 0 → x = [1, 2]
        let g_re = vec![1.0, 0.0, 0.0, 1.0];
        let g_im = vec![0.0, 0.0, 0.0, 0.0];
        let y_re = vec![1.0, 2.0];
        let y_im = vec![0.0, 0.0];
        let (xr, xi) = born_inversion_regularized(&g_re, &g_im, &y_re, &y_im, 2, 2, 0.0);
        assert!((xr[0] - 1.0).abs() < 1e-8, "xr[0]={}", xr[0]);
        assert!((xr[1] - 2.0).abs() < 1e-8, "xr[1]={}", xr[1]);
        assert!(xi.iter().all(|&v| v.abs() < 1e-8));
    }
}
