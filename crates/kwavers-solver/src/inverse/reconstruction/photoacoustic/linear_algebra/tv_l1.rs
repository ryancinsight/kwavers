//! Total Variation (ISTA) and L1/Lasso (FISTA) regularized solvers.

use ndarray::{Array1, Array2, ArrayView1};

use kwavers_core::error::KwaversResult;

use super::PhotoacousticLinearSolver;

impl PhotoacousticLinearSolver {
    /// Solve using Total Variation regularization.
    ///
    /// min ‖Ax − b‖² + λ TV(x)
    ///
    /// Uses the Iterative Shrinkage-Thresholding Algorithm (ISTA). The
    /// Lipschitz constant is estimated via power iteration.
    ///
    /// # Errors
    /// Returns `Err` when the RHS vector length does not match the matrix row
    /// count, or when the Lipschitz power iteration fails.
    pub fn solve_tv_regularized(
        &self,
        a: &Array2<f64>,
        b: ArrayView1<f64>,
        lambda: f64,
        shape: [usize; 3],
    ) -> KwaversResult<Array1<f64>> {
        let (m, n) = a.dim();

        if b.len() != m {
            return Err(kwavers_core::error::NumericalError::MatrixDimension {
                operation: "Total Variation regularized least squares".to_owned(),
                expected: format!("RHS vector length {} to match matrix rows {}", m, m),
                actual: format!("RHS vector length {}", b.len()),
            }
            .into());
        }

        let mut x = Array1::zeros(n);
        let at = a.t();
        let ata = at.dot(a);
        let atb = at.dot(&b);

        let lipschitz = self.power_method(&ata)?;
        let step_size = 1.0 / lipschitz;

        for _iter in 0..self.max_iterations {
            let gradient = ata.dot(&x) - &atb;
            let x_grad = &x - step_size * &gradient;
            x = self.tv_proximal(&x_grad, lambda * step_size, shape)?;

            let residual = a.dot(&x) - b;
            if residual.dot(&residual).sqrt() < self.tolerance {
                break;
            }
        }

        Ok(x)
    }

    /// Power method to estimate the largest eigenvalue of a symmetric matrix.
    ///
    /// Runs 100 iterations of `v ← Av / ‖Av‖` and returns `vᵀAv`.
    ///
    /// # Errors
    /// Always returns `Ok`; signature is `KwaversResult` for consistency with
    /// callers that propagate the error chain.
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

    /// Proximal operator for Total Variation (3-D finite-difference).
    ///
    /// Applies soft thresholding to each finite-difference gradient triplet
    /// `(Δx, Δy, Δz)` at every interior voxel.
    ///
    /// # Errors
    /// Always returns `Ok`; signature matches callers that propagate errors.
    fn tv_proximal(
        &self,
        x: &Array1<f64>,
        threshold: f64,
        shape: [usize; 3],
    ) -> KwaversResult<Array1<f64>> {
        let [nx, ny, nz] = shape;
        let mut result = x.clone();

        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                for k in 0..nz - 1 {
                    let idx = i * ny * nz + j * nz + k;
                    let idx_x = (i + 1) * ny * nz + j * nz + k;
                    let idx_y = i * ny * nz + (j + 1) * nz + k;
                    let idx_z = i * ny * nz + j * nz + (k + 1);

                    if idx_x < x.len() && idx_y < x.len() && idx_z < x.len() {
                        let grad_x = x[idx_x] - x[idx];
                        let grad_y = x[idx_y] - x[idx];
                        let grad_z = x[idx_z] - x[idx];
                        let grad_mag = grad_z
                            .mul_add(grad_z, grad_x.mul_add(grad_x, grad_y * grad_y))
                            .sqrt();

                        if grad_mag > threshold {
                            let factor = (grad_mag - threshold) / grad_mag;
                            result[idx] = factor
                                .mul_add(x[idx] - (x[idx_x] + x[idx_y] + x[idx_z]) / 3.0, x[idx]);
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}
