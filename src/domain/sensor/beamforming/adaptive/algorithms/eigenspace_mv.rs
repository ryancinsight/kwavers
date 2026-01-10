//! Eigenspace-based Minimum Variance (ESPMV) beamformer
//!
//! # SSOT / correctness stance (strict)
//! ESPMV requires a **complex Hermitian eigendecomposition** to construct a signal-subspace
//! projector `P_S`. Under strict SSOT, this module must route eigendecomposition through
//! `crate::utils::linear_algebra` and must not silently mask numerical failures.
//!
//! # Mathematical definition
//! Let `R_loaded = R + δ I` be (Hermitian) diagonally loaded covariance.
//! Let `V_s` be the signal-subspace basis consisting of the eigenvectors associated with the
//! largest `d` eigenvalues (where `d = num_sources`).
//!
//! A common ESPMV construction uses the projection `P_s = V_s V_sᴴ` and forms an MVDR-like solve
//! inside the projected space. Here we implement the canonical projected MVDR weight:
//!
//! `w = (R_loaded^{-1} P_s a) / (aᴴ P_s R_loaded^{-1} P_s a)`
//!
//! This preserves the MVDR unit-gain constraint (in the projected sense) and reduces sensitivity
//! to steering mismatch when the signal/noise subspace split is correct.
//!
//! # References
//! - Van Trees (2002), *Optimum Array Processing*
//! - Gershman et al. (1999), robust adaptive beamforming in subspaces
//! - Shahbazpanahi et al. (2003), generalized Capon / subspace localization

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

use super::BeamformingAlgorithm;

/// Eigenspace-based Minimum Variance (ESPMV) beamformer
#[derive(Debug)]
pub struct EigenspaceMV {
    /// Number of sources (signal subspace dimension)
    pub num_sources: usize,
    /// Diagonal loading factor
    pub diagonal_loading: f64,
}

impl EigenspaceMV {
    /// Create Eigenspace MV beamformer
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self {
            num_sources,
            diagonal_loading: 1e-6,
        }
    }

    /// Create with custom diagonal loading
    #[must_use]
    pub fn with_diagonal_loading(num_sources: usize, diagonal_loading: f64) -> Self {
        Self {
            num_sources,
            diagonal_loading,
        }
    }

    /// SSOT-correct ESPMV weight computation with explicit error handling.
    ///
    /// # Errors
    /// - If dimensions are inconsistent.
    /// - If `num_sources` is invalid (`0` or `> n`).
    /// - If diagonal loading is invalid (`NaN` or negative).
    /// - If SSOT eigendecomposition or SSOT complex solve fails.
    /// - If the normalization denominator is non-finite or non-positive.
    pub fn compute_weights_ssot(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = covariance.nrows();
        if n == 0 || covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(
                "EigenspaceMV::compute_weights_ssot: covariance must be non-empty and square"
                    .to_string(),
            ));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "EigenspaceMV::compute_weights_ssot: steering length ({}) must match covariance dimension ({n})",
                steering.len()
            )));
        }
        if self.num_sources == 0 || self.num_sources > n {
            return Err(KwaversError::InvalidInput(format!(
                "EigenspaceMV::compute_weights_ssot: num_sources ({}) must be in 1..={n}",
                self.num_sources
            )));
        }
        if !self.diagonal_loading.is_finite() || self.diagonal_loading < 0.0 {
            return Err(KwaversError::InvalidInput(
                "EigenspaceMV::compute_weights_ssot: diagonal_loading must be finite and >= 0"
                    .to_string(),
            ));
        }

        // R_loaded = R + δ I
        let mut r_loaded = covariance.clone();
        if self.diagonal_loading > 0.0 {
            for i in 0..n {
                r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
            }
        }

        // SSOT eigendecomposition requires `Complex<f64>` (num_complex::Complex).
        // Convert explicitly; no silent fallbacks.
        let r_loaded_ssot = r_loaded.mapv(|z| num_complex::Complex::<f64>::new(z.re, z.im));

        let (evals, evecs_ssot) =
            LinearAlgebra::hermitian_eigendecomposition_complex(&r_loaded_ssot)?;

        // Sort eigenpairs by descending eigenvalue (signal subspace = largest d).
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| {
            evals[j]
                .partial_cmp(&evals[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build V_s (n x d)
        let d = self.num_sources;
        let mut v_s = Array2::<Complex64>::zeros((n, d));
        for (col_out, &col_in) in idx.iter().take(d).enumerate() {
            for row in 0..n {
                let z = evecs_ssot[(row, col_in)];
                v_s[(row, col_out)] = Complex64::new(z.re, z.im);
            }
        }

        // P_s a = V_s (V_sᴴ a)
        let mut vhs_a = Array1::<Complex64>::zeros(d);
        for j in 0..d {
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..n {
                acc += v_s[(i, j)].conj() * steering[i];
            }
            vhs_a[j] = acc;
        }

        let mut p_s_a = Array1::<Complex64>::zeros(n);
        for i in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..d {
                acc += v_s[(i, j)] * vhs_a[j];
            }
            p_s_a[i] = acc;
        }

        // Solve R_loaded y = P_s a (SSOT complex solve; avoid explicit inverse).
        let r_loaded_solve = r_loaded.mapv(|z| num_complex::Complex::<f64>::new(z.re, z.im));
        let p_s_a_solve = p_s_a.mapv(|z| num_complex::Complex::<f64>::new(z.re, z.im));
        let y_ssot = LinearAlgebra::solve_linear_system_complex(&r_loaded_solve, &p_s_a_solve)?;
        let y = y_ssot.mapv(|z| Complex64::new(z.re, z.im));

        // Apply P_s again: P_s y
        let mut vhs_y = Array1::<Complex64>::zeros(d);
        for j in 0..d {
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..n {
                acc += v_s[(i, j)].conj() * y[i];
            }
            vhs_y[j] = acc;
        }
        let mut p_s_y = Array1::<Complex64>::zeros(n);
        for i in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..d {
                acc += v_s[(i, j)] * vhs_y[j];
            }
            p_s_y[i] = acc;
        }

        // denom = aᴴ (P_s y)  (should be real-positive for well-posed covariance + loading)
        let denom: Complex64 = steering
            .iter()
            .zip(p_s_y.iter())
            .map(|(a, zi)| a.conj() * zi)
            .sum();

        let denom_re = denom.re;
        if !denom_re.is_finite() || denom_re <= 0.0 {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::InvalidOperation(
                    "EigenspaceMV::compute_weights_ssot: non-positive or non-finite denominator a^H P_s R^{-1} P_s a".to_string(),
                ),
            ));
        }

        // w = (P_s y) / denom
        Ok(p_s_y.mapv(|z| z / denom_re))
    }
}

impl BeamformingAlgorithm for EigenspaceMV {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        self.compute_weights_ssot(covariance, steering)
    }
}
