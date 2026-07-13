//! Total Variation (ISTA) and L1/Lasso (FISTA) regularized solvers.

use leto::{Array1, Array2, ArrayView1};

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
        let [m, n] = a.shape();

        if b.shape()[0] != m {
            return Err(kwavers_core::error::NumericalError::MatrixDimension {
                operation: "Total Variation regularized least squares".to_owned(),
                expected: format!("RHS vector length {} to match matrix rows {}", m, m),
                actual: format!("RHS vector length {}", b.shape()[0]),
            }
            .into());
        }

        let mut x = Array1::<f64>::zeros(n);
        // AᵀA (n×n) and Aᵀb (n) via native leto linear algebra.
        let at = a.transpose([1, 0]).expect("invariant: TV transpose valid");
        let mut ata = Array2::<f64>::zeros((n, n));
        leto_ops::matmul(&at, &a.view(), &mut ata.view_mut()).expect("invariant: TV AᵀA conforms");
        let mut atb = Array1::<f64>::zeros(n);
        leto_ops::matvec(&at, &b, &mut atb.view_mut()).expect("invariant: TV Aᵀb conforms");

        let lipschitz = self.power_method(&ata)?;
        let step_size = 1.0 / lipschitz;

        for _iter in 0..self.max_iterations {
            // gradient = AᵀA·x − Aᵀb
            let mut gradient = Array1::<f64>::zeros(n);
            leto_ops::matvec(&ata.view(), &x.view(), &mut gradient.view_mut())
                .expect("invariant: TV AᵀA·x conforms");
            for j in 0..n {
                gradient[j] -= atb[j];
            }
            // x_grad = x − step·gradient
            let mut x_grad = Array1::<f64>::zeros(n);
            for j in 0..n {
                x_grad[j] = x[j] - step_size * gradient[j];
            }
            x = self.tv_proximal(&x_grad, lambda * step_size, shape)?;

            // residual = A·x − b
            let mut residual = Array1::<f64>::zeros(m);
            leto_ops::matvec(&a.view(), &x.view(), &mut residual.view_mut())
                .expect("invariant: TV A·x conforms");
            for i in 0..m {
                residual[i] -= b[i];
            }
            if leto_ops::dot(&residual.view(), &residual.view())
                .expect("invariant: TV residual norm conforms")
                .sqrt()
                < self.tolerance
            {
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
        let n = a.shape()[0];
        let mut v = Array1::<f64>::ones(n);
        let norm0 = leto_ops::dot(&v.view(), &v.view())
            .expect("invariant: power-method ‖v₀‖ conforms")
            .sqrt();
        for j in 0..n {
            v[j] /= norm0;
        }

        let mut eigenvalue = 0.0;
        for _ in 0..100 {
            let mut av = Array1::<f64>::zeros(n);
            leto_ops::matvec(&a.view(), &v.view(), &mut av.view_mut())
                .expect("invariant: power-method A·v conforms");
            eigenvalue = leto_ops::dot(&v.view(), &av.view())
                .expect("invariant: power-method vᵀAv conforms");
            let av_norm = leto_ops::dot(&av.view(), &av.view())
                .expect("invariant: power-method ‖Av‖ conforms")
                .sqrt();
            for j in 0..n {
                v[j] = av[j] / av_norm;
            }
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

                    if idx_x < (x.len()) && idx_y < (x.len()) && idx_z < (x.len()) {
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
