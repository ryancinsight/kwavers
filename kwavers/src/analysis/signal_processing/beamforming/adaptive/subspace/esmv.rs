//! Eigenspace Minimum Variance (ESMV) beamformer.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

/// Eigenspace Minimum Variance (ESMV) Beamformer
///
/// Constrains adaptive beamforming to the signal subspace for robustness
/// against noise and interference.
///
/// # Mathematical Definition
///
/// ```text
/// w = P_s R^{-1} a / (a^H R^{-1} P_s a)
/// ```
///
/// where **P_s = E_s E_s^H** is the signal subspace projector.
///
/// # References
///
/// - Gershman et al. (1999), "Adaptive beamforming algorithms with robustness
///   against jammer motion"
#[derive(Debug, Clone, Copy)]
pub struct EigenspaceMV {
    /// Number of sources (signal subspace dimension M)
    pub num_sources: usize,
    /// Diagonal loading factor (for numerical stability)
    pub diagonal_loading: f64,
}

impl EigenspaceMV {
    /// Create ESMV beamformer with default diagonal loading.
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self {
            num_sources,
            diagonal_loading: 1e-6,
        }
    }

    /// Create ESMV beamformer with custom diagonal loading.
    #[must_use]
    pub fn with_diagonal_loading(num_sources: usize, diagonal_loading: f64) -> Self {
        Self {
            num_sources,
            diagonal_loading,
        }
    }

    /// Compute ESMV beamforming weights.
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// w = P_s R^{-1} a / (a^H R^{-1} P_s a)
    /// ```
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = covariance.nrows();

        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(
                "EigenspaceMV::compute_weights: covariance must be non-empty square matrix"
                    .to_owned(),
            ));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "EigenspaceMV::compute_weights: steering vector length {} does not match covariance dimension {}",
                steering.len(),
                n
            )));
        }
        if self.num_sources >= n {
            return Err(KwaversError::InvalidInput(format!(
                "EigenspaceMV::compute_weights: num_sources {} must be < N {}",
                self.num_sources, n
            )));
        }

        for &val in steering {
            if !val.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::NaN {
                    operation: "EigenspaceMV::compute_weights".to_owned(),
                    inputs: "steering vector contains non-finite values".to_owned(),
                }));
            }
        }

        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
        }

        let r_for_eig = r_loaded.mapv(|z| num_complex::Complex::new(z.re, z.im));
        let (eigenvalues, eigenvectors) =
            LinearAlgebra::hermitian_eigendecomposition_complex(&r_for_eig)?;

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut p_s = Array2::<Complex64>::zeros((n, n));
        for &idx in indices.iter().take(self.num_sources) {
            let eigenvec = eigenvectors.slice(s![.., idx]);

            for i in 0..n {
                for j in 0..n {
                    let e_i = eigenvec[i];
                    let e_j = eigenvec[j];
                    let e_i_c64 = Complex64::new(e_i.re, e_i.im);
                    let e_j_c64 = Complex64::new(e_j.re, e_j.im);
                    p_s[(i, j)] += e_i_c64 * e_j_c64.conj();
                }
            }
        }

        let r_for_solve = r_loaded.mapv(|z| num_complex::Complex::new(z.re, z.im));
        let a_for_solve = steering.mapv(|z| num_complex::Complex::new(z.re, z.im));
        let r_inv_a_raw = LinearAlgebra::solve_linear_system_complex(&r_for_solve, &a_for_solve)?;
        let r_inv_a = r_inv_a_raw.mapv(|z| Complex64::new(z.re, z.im));

        let mut ps_r_inv_a = Array1::<Complex64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                ps_r_inv_a[i] += p_s[(i, j)] * r_inv_a[j];
            }
        }

        let mut a_h_r_inv_ps_a = Complex64::zero();
        for i in 0..n {
            a_h_r_inv_ps_a += steering[i].conj() * ps_r_inv_a[i];
        }

        if a_h_r_inv_ps_a.norm() < 1e-12 {
            return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                "EigenspaceMV::compute_weights: denominator near zero (signal subspace mismatch)"
                    .to_owned(),
            )));
        }

        let weights = ps_r_inv_a.mapv(|x| x / a_h_r_inv_ps_a);

        for &w in &weights {
            if !w.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    "EigenspaceMV::compute_weights: non-finite weight computed".to_owned(),
                )));
            }
        }

        Ok(weights)
    }
}
