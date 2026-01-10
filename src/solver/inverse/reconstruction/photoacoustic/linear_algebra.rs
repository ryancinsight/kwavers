//! Robust linear algebra solvers for photoacoustic reconstruction
//!
//! This module provides efficient and numerically stable linear algebra
//! operations for solving the inverse problems in photoacoustic imaging.

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2, ArrayView1};

/// Linear algebra solver with various regularization methods
#[derive(Debug)]
pub struct LinearSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl LinearSolver {
    /// Create new linear solver
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }

    /// Solve regularized least squares: min ||Ax - b||² + λ||Lx||²
    ///
    /// Uses Conjugate Gradient for Normal Equations (CGNE) with Tikhonov regularization
    #[allow(dead_code)]
    pub fn solve_tikhonov(
        &self,
        a: &Array2<f64>,
        b: ArrayView1<f64>,
        lambda: f64,
        l: Option<&Array2<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        let (m, n) = a.dim();

        // Validate dimensions
        if b.len() != m {
            return Err(
                crate::domain::core::error::NumericalError::MatrixDimension {
                    operation: "Tikhonov regularized least squares".to_string(),
                    expected: format!("RHS vector length {} to match matrix rows {}", m, m),
                    actual: format!("RHS vector length {}", b.len()),
                }
                .into(),
            );
        }

        // Form normal equations: (A^T A + λ L^T L) x = A^T b
        let at = a.t();
        let atb = at.dot(&b);

        // Regularization matrix (identity if not provided)
        let regularizer = if let Some(l_mat) = l {
            l_mat.t().dot(l_mat)
        } else {
            Array2::eye(n)
        };

        // System matrix: A^T A + λ L^T L
        let mut system_matrix = at.dot(a);
        system_matrix.scaled_add(lambda, &regularizer);

        // Solve using Conjugate Gradient
        self.conjugate_gradient(&system_matrix, &atb)
    }

    /// Conjugate Gradient solver for symmetric positive definite systems
    #[allow(dead_code)]
    pub fn conjugate_gradient(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let n = b.len();
        let mut x = Array1::zeros(n);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for iter in 0..self.max_iterations {
            let ap = a.dot(&p);
            let alpha = rsold / p.dot(&ap);

            x.scaled_add(alpha, &p);
            r.scaled_add(-alpha, &ap);

            let rsnew = r.dot(&r);

            // Check convergence
            if rsnew.sqrt() < self.tolerance {
                log::debug!("Conjugate gradient converged after {} iterations", iter + 1);
                return Ok(x);
            }

            let beta = rsnew / rsold;
            p = &r + beta * &p;
            rsold = rsnew;
        }

        // If we're here, we didn't converge
        Err(
            crate::domain::core::error::NumericalError::ConvergenceFailed {
                method: "Conjugate Gradient".to_string(),
                iterations: self.max_iterations,
                error: rsold.sqrt(),
            }
            .into(),
        )
    }

    /// Solve using Total Variation regularization
    ///
    /// min ||Ax - b||² + λ TV(x)
    /// where TV(x) is the total variation (L1 norm of gradient)
    pub fn solve_tv_regularized(
        &self,
        a: &Array2<f64>,
        b: ArrayView1<f64>,
        lambda: f64,
        shape: [usize; 3],
    ) -> KwaversResult<Array1<f64>> {
        let (m, n) = a.dim();

        // Validate dimensions
        if b.len() != m {
            return Err(
                crate::domain::core::error::NumericalError::MatrixDimension {
                    operation: "Total Variation regularized least squares".to_string(),
                    expected: format!("RHS vector length {} to match matrix rows {}", m, m),
                    actual: format!("RHS vector length {}", b.len()),
                }
                .into(),
            );
        }

        let mut x = Array1::zeros(n);

        // Use Iterative Shrinkage-Thresholding Algorithm (ISTA)
        let at = a.t();
        let ata = at.dot(a);
        let atb = at.dot(&b);

        // Estimate Lipschitz constant (largest eigenvalue of A^T A)
        let lipschitz = self.power_method(&ata)?;
        let step_size = 1.0 / lipschitz;

        for _iter in 0..self.max_iterations {
            // Gradient step
            let gradient = ata.dot(&x) - &atb;
            let x_grad = &x - step_size * &gradient;

            // Proximal operator for TV (soft thresholding on gradient)
            x = self.tv_proximal(&x_grad, lambda * step_size, shape)?;

            // Check convergence
            let residual = a.dot(&x) - b;
            if residual.dot(&residual).sqrt() < self.tolerance {
                break;
            }
        }

        Ok(x)
    }

    /// Power method to estimate largest eigenvalue
    fn power_method(&self, a: &Array2<f64>) -> KwaversResult<f64> {
        let n = a.nrows();
        let mut v = Array1::<f64>::ones(n);
        v /= v.dot(&v).sqrt();

        let mut eigenvalue = 0.0;

        for _ in 0..100 {
            let av = a.dot(&v);
            eigenvalue = v.dot(&av);
            let av_norm = av.dot(&av).sqrt();
            v = av / av_norm;
        }

        Ok(eigenvalue)
    }

    /// Proximal operator for Total Variation
    fn tv_proximal(
        &self,
        x: &Array1<f64>,
        threshold: f64,
        shape: [usize; 3],
    ) -> KwaversResult<Array1<f64>> {
        let [nx, ny, nz] = shape;
        let mut result = x.clone();

        // Apply soft thresholding to gradients
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                for k in 0..nz - 1 {
                    let idx = i * ny * nz + j * nz + k;
                    let idx_x = (i + 1) * ny * nz + j * nz + k;
                    let idx_y = i * ny * nz + (j + 1) * nz + k;
                    let idx_z = i * ny * nz + j * nz + (k + 1);

                    if idx_x < x.len() && idx_y < x.len() && idx_z < x.len() {
                        // Compute gradient magnitude
                        let grad_x = x[idx_x] - x[idx];
                        let grad_y = x[idx_y] - x[idx];
                        let grad_z = x[idx_z] - x[idx];
                        let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();

                        // Soft thresholding
                        if grad_mag > threshold {
                            let factor = (grad_mag - threshold) / grad_mag;
                            result[idx] =
                                x[idx] + factor * (x[idx] - (x[idx_x] + x[idx_y] + x[idx_z]) / 3.0);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Solve using L1 regularization (Lasso)
    ///
    /// min ||Ax - b||² + λ||x||₁
    #[allow(dead_code)]
    pub fn solve_l1_regularized(
        &self,
        a: &Array2<f64>,
        b: ArrayView1<f64>,
        lambda: f64,
    ) -> KwaversResult<Array1<f64>> {
        let (m, n) = a.dim();

        // Validate dimensions
        if b.len() != m {
            return Err(
                crate::domain::core::error::NumericalError::MatrixDimension {
                    operation: "L1 regularized least squares (Lasso)".to_string(),
                    expected: format!("RHS vector length {} to match matrix rows {}", m, m),
                    actual: format!("RHS vector length {}", b.len()),
                }
                .into(),
            );
        }

        let mut x = Array1::zeros(n);

        // Use Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
        let at = a.t();
        let ata = at.dot(a);
        let atb = at.dot(&b);

        // Estimate Lipschitz constant
        let lipschitz = self.power_method(&ata)?;
        let step_size = 1.0 / lipschitz;

        let mut y = x.clone();
        let mut t = 1.0;

        for _iter in 0..self.max_iterations {
            let x_previous = x.clone();

            // Gradient step
            let gradient = ata.dot(&y) - &atb;
            let x_grad = &y - step_size * &gradient;

            // Soft thresholding (proximal operator for L1)
            x = self.soft_threshold(&x_grad, lambda * step_size);

            // FISTA momentum update
            let t_next = (1.0 + (1.0_f64 + 4.0 * t * t).sqrt()) / 2.0;
            y = &x + ((t - 1.0) / t_next) * (&x - &x_previous);
            t = t_next;

            // Check convergence
            let residual = a.dot(&x) - b;
            if residual.dot(&residual).sqrt() < self.tolerance {
                break;
            }
        }

        Ok(x)
    }

    /// Soft thresholding operator
    #[allow(dead_code)]
    fn soft_threshold(&self, x: &Array1<f64>, threshold: f64) -> Array1<f64> {
        x.mapv(|xi| {
            if xi > threshold {
                xi - threshold
            } else if xi < -threshold {
                xi + threshold
            } else {
                0.0
            }
        })
    }

    /// Truncated SVD for ill-conditioned problems
    pub fn solve_truncated_svd(
        &self,
        a: &Array2<f64>,
        b: ArrayView1<f64>,
        truncation: f64,
    ) -> KwaversResult<Array1<f64>> {
        // Compute SVD using power iteration method
        let (u, s, vt) = self.power_iteration_svd(a)?;

        // Truncate small singular values
        let s_max = s.iter().copied().fold(0.0, f64::max);
        let threshold = truncation * s_max;

        // Compute pseudoinverse solution
        let mut x = Array1::zeros(vt.nrows());
        for (i, &s_val) in s.iter().enumerate() {
            if s_val > threshold {
                let ui = u.column(i);
                let vi = vt.row(i);
                x += &(vi.to_owned() * (ui.dot(&b) / s_val));
            }
        }

        Ok(x)
    }

    /// Power iteration-based SVD implementation
    ///
    /// Uses power iteration method to compute singular value decomposition.
    /// For production use, consider using a more robust algorithm like LAPACK's DGESVD.
    fn power_iteration_svd(
        &self,
        a: &Array2<f64>,
    ) -> KwaversResult<(Array2<f64>, Vec<f64>, Array2<f64>)> {
        // Power iteration method for SVD computation
        let (m, n) = a.dim();
        let k = m.min(n);

        // Compute A^T A
        let ata = a.t().dot(a);

        // Power iteration to find eigenvectors
        let mut v = Array2::eye(n);
        let mut s = vec![0.0; k];

        for i in 0..k {
            // Find dominant eigenvector
            let mut vi = Array1::ones(n);
            for _ in 0..100 {
                vi = ata.dot(&vi);
                vi /= vi.dot(&vi).sqrt();
            }

            // Compute singular value
            let avi = a.dot(&vi);
            s[i] = avi.dot(&avi).sqrt();

            // Store in V
            for j in 0..n {
                v[[j, i]] = vi[j];
            }
        }

        // Compute U = AV/S
        let mut u = Array2::zeros((m, k));
        for i in 0..k {
            if s[i] > 1e-10 {
                let vi = v.column(i);
                let ui = a.dot(&vi) / s[i];
                for j in 0..m {
                    u[[j, i]] = ui[j];
                }
            }
        }

        Ok((u, s, v.t().to_owned()))
    }
}
