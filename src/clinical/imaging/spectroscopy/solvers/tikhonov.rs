//! Tikhonov-regularized Least Squares Solver
//!
//! Provides mathematically rigorous solutions for potentially ill-conditioned
//! linear systems in spectroscopic unmixing.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};

/// Solve Tikhonov-regularized least squares: (EᵀE + λI)C = Eᵀμ
///
/// # Arguments
///
/// - `e`: Extinction matrix (m × n)
/// - `mu`: Absorption spectrum (m × 1)
/// - `lambda`: Regularization parameter (Tikhonov parameter)
///
/// # Returns
///
/// Concentration vector C (n × 1)
#[allow(non_snake_case)] // E is standard notation for extinction coefficient matrix
pub fn tikhonov_solve(e: &Array2<f64>, mu: &Array1<f64>, lambda: f64) -> Result<Array1<f64>> {
    let n_chromophores = e.ncols();

    // Compute EᵀE (Gram matrix)
    let et = e.t();
    let ete = et.dot(e);

    // Add regularization: EᵀE + λI
    // This ensures the matrix is positive-definite even if E is rank-deficient
    let mut ete_reg = ete.clone();
    for i in 0..n_chromophores {
        ete_reg[[i, i]] += lambda.max(1e-12);
    }

    // Compute Eᵀμ (Right-hand side)
    let et_mu = et.dot(mu);

    // Solve (EᵀE + λI)C = Eᵀμ using Cholesky decomposition (LLᵀ)
    // Cholesky is more stable and efficient for SPD systems than Gaussian elimination
    cholesky_solve(&ete_reg, &et_mu)
        .context("Failed to solve Tikhonov system via Cholesky decomposition")
}

/// Solve symmetric positive-definite system Ax = b using Cholesky decomposition (LLᵀ)
///
/// # Invariants
/// - A must be symmetric and positive-definite
/// - Smallest eigenvalue of A must be significantly larger than machine epsilon
#[allow(non_snake_case)]
fn cholesky_solve(A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = A.nrows();
    if A.ncols() != n {
        anyhow::bail!("Matrix A must be square");
    }

    // 1. Compute Cholesky Factor L where A = LLᵀ
    // Using the Cholesky-Banachiewicz algorithm
    let mut L = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += L[[i, k]] * L[[j, k]];
            }

            if i == j {
                let diag_val = A[[i, i]] - sum;
                if diag_val <= 0.0 {
                    anyhow::bail!("Matrix is not positive-definite (diag value: {})", diag_val);
                }
                L[[i, j]] = diag_val.sqrt();
            } else {
                if L[[j, j]].abs() < 1e-15 {
                    anyhow::bail!("Matrix is singular or near-singular in Cholesky decomposition");
                }
                L[[i, j]] = (A[[i, j]] - sum) / L[[j, j]];
            }
        }
    }

    // 2. Solve Ly = b (Forward substitution)
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += L[[i, j]] * y[j];
        }
        y[i] = (b[i] - sum) / L[[i, i]];
    }

    // 3. Solve Lᵀx = y (Backward substitution)
    let mut x = Array1::zeros(n);
    let LT = L.t();
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += LT[[i, j]] * x[j];
        }
        x[i] = (y[i] - sum) / LT[[i, i]];
    }

    Ok(x)
}

/// Estimate condition number (max/min singular value ratio)
#[allow(non_snake_case)]
pub fn estimate_condition_number(A: &Array2<f64>) -> Result<f64> {
    let (_, n) = A.dim();
    let ata = A.t().dot(A);

    // Power iteration for largest eigenvalue λ_max
    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    for _ in 0..20 {
        let v_new = ata.dot(&v);
        let norm = v_new.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        v = v_new / norm;
    }
    let lambda_max = v.dot(&ata.dot(&v));

    // Approximate smallest eigenvalue λ_min via Trace(AᵀA) - λ_max (rough heuristic for small n)
    // For n=2,3 this is acceptable. For larger n, we would need inverse power iteration.
    let trace: f64 = (0..n).map(|i| ata[[i, i]]).sum();
    let lambda_min = (trace - lambda_max).max(1e-15) / (n as f64 - 1.0).max(1.0);

    let condition = (lambda_max / lambda_min).sqrt();
    Ok(condition)
}
