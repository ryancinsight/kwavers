//! BiCGSTAB (real and complex) implementations for [`IterativeSolver`].
//!
//! References:
//! - van der Vorst (1992): "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG"
//! - Sleijpen & Fokkema (1993): complex extension

use super::super::csr::CompressedSparseRowMatrix;
use super::IterativeSolver;
use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{Array1, ArrayView1};

impl IterativeSolver {
    fn dot_real(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// BiCGSTAB for real non-symmetric sparse systems.
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the method fails to converge.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn bicgstab_real(
        &self,
        a: &CompressedSparseRowMatrix,
        b: ArrayView1<f64>,
        x0: Option<ArrayView1<f64>>,
    ) -> KwaversResult<Array1<f64>> {
        let n = a.rows;
        let mut x = if let Some(v) = x0 {
            Array1::from_shape_vec([n], v.iter().copied().collect())
                .expect("initial guess length must match solver dimension")
        } else {
            Array1::zeros([n])
        };

        let b_owned = Array1::from_shape_vec([n], b.iter().copied().collect())
            .expect("rhs length must match solver dimension");
        let mut r = &b_owned - &a.multiply_vector(x.view())?;
        let r0 = r.clone();
        let initial_residual = Self::dot_real(&r, &r).sqrt();
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
            rho = Self::dot_real(&r0, &r);

            if rho.abs() < 1e-14 {
                if self.config.verbose {
                    log::info!("BiCGSTAB converged in {} iterations", iteration);
                }
                break;
            }

            let beta = (rho / rho_prev) * (alpha / omega);
            let omega_v = &v * omega;
            let direction = &p - &omega_v;
            p = &r + &(&direction * beta);

            v = a.multiply_vector(p.view())?;
            alpha = rho / Self::dot_real(&r0, &v);

            let s = &r - &(&v * alpha);

            if Self::dot_real(&s, &s).sqrt() < self.config.tolerance {
                x = &x + &(&p * alpha);
                return Ok(x);
            }

            let t = a.multiply_vector(s.view())?;
            omega = Self::dot_real(&t, &s) / Self::dot_real(&t, &t);

            let alpha_p = &p * alpha;
            let omega_s = &s * omega;
            let update = &alpha_p + &omega_s;
            x = &x + &update;
            r = &s - &(&t * omega);

            let residual_norm = Self::dot_real(&r, &r).sqrt();
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

        let final_residual = Self::dot_real(&r, &r).sqrt();
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
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn bicgstab_complex_impl(
        &self,
        a: &CompressedSparseRowMatrix<Complex64>,
        b: ArrayView1<Complex64>,
        x0: Option<ArrayView1<Complex64>>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = a.rows;
        let mut x = if let Some(v) = x0 {
            Array1::from_shape_vec([n], v.iter().copied().collect())
                .expect("initial complex guess length must match solver dimension")
        } else {
            Array1::from_elem([n], Complex64::default())
        };

        let b_owned = Array1::from_shape_vec([n], b.iter().copied().collect())
            .expect("complex rhs length must match solver dimension");
        let mut r = &b_owned - &a.multiply_vector(x.view())?;
        let r0 = r.clone();
        let initial_residual = r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if initial_residual < self.config.tolerance {
            return Ok(x);
        }

        let mut rho = Complex64::new(1.0, 0.0);
        let mut alpha = Complex64::new(1.0, 0.0);
        let mut omega = Complex64::new(1.0, 0.0);

        let mut v = Array1::from_elem([n], Complex64::default());
        let mut p = Array1::from_elem([n], Complex64::default());

        for iteration in 0..self.config.max_iterations {
            let rho_prev = rho;
            rho = r0
                .iter()
                .zip(r.iter())
                .map(|(a, b)| a.conj() * b)
                .sum::<Complex64>();

            if rho.norm() < 1e-14 {
                let residual_norm = r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if residual_norm < self.config.tolerance {
                    return Ok(x);
                }
                if self.config.verbose {
                    log::info!("BiCGSTAB (Complex) breakdown in {} iterations", iteration);
                }
                break;
            }

            let beta = (rho / rho_prev) * (alpha / omega);
            let omega_v = v.mapv(|value| value * omega);
            let direction = &p - &omega_v;
            let beta_direction = direction.mapv(|value| value * beta);
            p = &r + &beta_direction;

            v = a.multiply_vector(p.view())?;

            let r0_v = r0
                .iter()
                .zip(v.iter())
                .map(|(a, b)| a.conj() * b)
                .sum::<Complex64>();
            alpha = if r0_v.norm() < 1e-14 {
                Complex64::new(1.0, 0.0)
            } else {
                rho / r0_v
            };

            let alpha_v = v.mapv(|value| value * alpha);
            let s = &r - &alpha_v;

            let s_norm = s.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if s_norm < self.config.tolerance {
                let alpha_p = p.mapv(|value| value * alpha);
                x = &x + &alpha_p;
                return Ok(x);
            }

            let t = a.multiply_vector(s.view())?;

            let t_norm_sqr = t.iter().map(|c| c.norm_sqr()).sum::<f64>();
            let t_s_dot = t
                .iter()
                .zip(s.iter())
                .map(|(ti, si)| ti.conj() * si)
                .sum::<Complex64>();

            omega = if t_norm_sqr < 1e-14 {
                Complex64::new(0.0, 0.0)
            } else {
                t_s_dot / t_norm_sqr
            };

            let alpha_p = p.mapv(|value| value * alpha);
            let omega_s = s.mapv(|value| value * omega);
            let update = &alpha_p + &omega_s;
            x = &x + &update;
            let omega_t = t.mapv(|value| value * omega);
            r = &s - &omega_t;

            let residual_norm = r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
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

        let final_residual = r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
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
