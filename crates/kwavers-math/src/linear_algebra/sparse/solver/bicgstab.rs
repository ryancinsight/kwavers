//! BiCGSTAB (real and complex) implementations for [`IterativeSolver`].
//!
//! References:
//! - van der Vorst (1992): "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG"
//! - Sleijpen & Fokkema (1993): complex extension

use super::super::csr::CompressedSparseRowMatrix;
use super::{axpy, conj_dot_complex, dot, norm_sqr_complex, scale_inplace, IterativeSolver};
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::Array1;
use num_complex::Complex64;

impl IterativeSolver {
    /// BiCGSTAB for real non-symmetric sparse systems.
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the method fails to converge.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn bicgstab_real(
        &self,
        a: &CompressedSparseRowMatrix,
        b: &Array1<f64>,
        x0: Option<&Array1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        let n = a.rows;
        let mut x = x0.map_or_else(|| Array1::zeros([n]), |v| v.clone());

        let mut r = {
            let mut tmp = b.clone();
            let ax = a.multiply_vector(&x)?;
            for i in 0..n {
                tmp[[i]] -= ax[[i]];
            }
            tmp
        };
        let r0 = r.clone();
        let initial_residual = dot(&r, &r).sqrt();
        if initial_residual < self.config.tolerance {
            return Ok(x);
        }

        let mut rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;

        let mut v = Array1::zeros([n]);
        let mut p = Array1::zeros([n]);

        for iteration in 0..self.config.max_iterations {
            let rho_prev = rho;
            rho = dot(&r0, &r);

            if rho.abs() < 1e-14 {
                if self.config.verbose {
                    log::info!("BiCGSTAB converged in {} iterations", iteration);
                }
                break;
            }

            let beta = (rho / rho_prev) * (alpha / omega);

            // p = r + beta * (p - omega * v)
            axpy(-omega, &v, &mut p);
            scale_inplace(&mut p, beta);
            axpy(1.0, &r, &mut p);

            v = a.multiply_vector(&p)?;
            alpha = rho / dot(&r0, &v);

            // s = r - alpha * v
            let mut s = r.clone();
            axpy(-alpha, &v, &mut s);

            if dot(&s, &s).sqrt() < self.config.tolerance {
                axpy(alpha, &p, &mut x);
                return Ok(x);
            }

            let t = a.multiply_vector(&s)?;
            omega = dot(&t, &s) / dot(&t, &t);

            axpy(alpha, &p, &mut x);
            axpy(omega, &s, &mut x);

            let mut r_new = s.clone();
            axpy(-omega, &t, &mut r_new);
            r = r_new;

            let residual_norm = dot(&r, &r).sqrt();
            if residual_norm < self.config.tolerance {
                if self.config.verbose {
                    log::info!(
                        "BiCGSTAB converged in {} iterations, residual: {:.2e}",
                        iteration + 1,
                        residual_norm
                    );
                }
                return Ok(x);
            }
        }

        let final_residual = dot(&r, &r).sqrt();
        if self.config.verbose {
            log::warn!(
                "BiCGSTAB failed to converge after {} iterations, residual: {:.2e}",
                self.config.max_iterations,
                final_residual
            );
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "bicgstab".to_owned(),
            iterations: self.config.max_iterations,
            error: final_residual,
        }))
    }

    /// BiCGSTAB for complex non-symmetric sparse systems.
    ///
    /// Uses conjugated inner products (r₀ᴴ·r) for correct BiCG orthogonality.
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the method fails to converge.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn bicgstab_complex_impl(
        &self,
        a: &CompressedSparseRowMatrix<Complex64>,
        b: &Array1<Complex64>,
        x0: Option<&Array1<Complex64>>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = a.rows;
        let mut x = x0.map_or_else(
            || Array1::from_shape_fn([n], |_| Complex64::default()),
            |v| v.clone(),
        );

        let mut r = {
            let mut tmp = b.clone();
            let ax = a.multiply_vector(&x)?;
            for i in 0..n {
                tmp[[i]] -= ax[[i]];
            }
            tmp
        };
        let r0 = r.clone();
        let initial_residual = norm_sqr_complex(&r).sqrt();
        if initial_residual < self.config.tolerance {
            return Ok(x);
        }

        let mut rho = Complex64::new(1.0, 0.0);
        let mut alpha = Complex64::new(1.0, 0.0);
        let mut omega = Complex64::new(1.0, 0.0);

        let mut v = Array1::from_shape_fn([n], |_| Complex64::default());
        let mut p = Array1::from_shape_fn([n], |_| Complex64::default());

        for iteration in 0..self.config.max_iterations {
            let rho_prev = rho;
            rho = conj_dot_complex(&r0, &r);

            if rho.norm() < 1e-14 {
                let residual_norm = norm_sqr_complex(&r).sqrt();
                if residual_norm < self.config.tolerance {
                    return Ok(x);
                }
                if self.config.verbose {
                    log::info!("BiCGSTAB (Complex) breakdown in {} iterations", iteration);
                }
                break;
            }

            let beta = (rho / rho_prev) * (alpha / omega);

            // p = r + beta * (p - omega * v)
            for i in 0..n {
                p[[i]] = r[[i]] + beta * (p[[i]] - omega * v[[i]]);
            }

            v = a.multiply_vector(&p)?;

            let r0_v = conj_dot_complex(&r0, &v);
            alpha = if r0_v.norm() < 1e-14 {
                Complex64::new(1.0, 0.0)
            } else {
                rho / r0_v
            };

            // s = r - alpha * v
            let mut s = r.clone();
            for i in 0..n {
                s[[i]] -= alpha * v[[i]];
            }

            let s_norm = norm_sqr_complex(&s).sqrt();
            if s_norm < self.config.tolerance {
                for i in 0..n {
                    x[[i]] += alpha * p[[i]];
                }
                return Ok(x);
            }

            let t = a.multiply_vector(&s)?;

            let t_norm_sqr = norm_sqr_complex(&t);
            let t_s_dot = conj_dot_complex(&t, &s);

            omega = if t_norm_sqr < 1e-14 {
                Complex64::new(0.0, 0.0)
            } else {
                t_s_dot / t_norm_sqr
            };

            for i in 0..n {
                x[[i]] = x[[i]] + alpha * p[[i]] + omega * s[[i]];
            }
            for i in 0..n {
                r[[i]] = s[[i]] - omega * t[[i]];
            }

            let residual_norm = norm_sqr_complex(&r).sqrt();
            if residual_norm < self.config.tolerance {
                if self.config.verbose {
                    log::info!(
                        "BiCGSTAB (Complex) converged in {} iterations, residual: {:.2e}",
                        iteration + 1,
                        residual_norm
                    );
                }
                return Ok(x);
            }
        }

        let final_residual = norm_sqr_complex(&r).sqrt();
        if self.config.verbose {
            log::warn!(
                "BiCGSTAB (Complex) failed to converge after {} iterations, residual: {:.2e}",
                self.config.max_iterations,
                final_residual
            );
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "bicgstab_complex".to_owned(),
            iterations: self.config.max_iterations,
            error: final_residual,
        }))
    }
}
