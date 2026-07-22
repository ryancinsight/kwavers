//! Eigenspace Minimum Variance (ESMV) beamformer.

use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use kwavers_math::linear_algebra::eigendecomposition::{EigenSolver, EigenSolverConfig};
use leto::{Array1, Array2, SliceArg};

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
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        let n = covariance.shape()[0];

        if n == 0 || covariance.shape()[1] != n {
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

        for &val in steering.iter() {
            if !val.re.is_finite() || !val.im.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::NaN {
                    operation: "EigenspaceMV::compute_weights".to_owned(),
                    inputs: "steering vector contains non-finite values".to_owned(),
                }));
            }
        }

        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[[i, i]] += Complex64::new(self.diagonal_loading, 0.0);
        }

        let (eigenvalues, eigenvectors) = {
            let r = EigenSolver::jacobi_hermitian(&r_loaded, EigenSolverConfig::default())?;
            (r.eigenvalues, r.eigenvectors)
        };

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));

        let mut p_s = Array2::<Complex64>::from_elem((n, n), Complex64::default());
        for &idx in indices.iter().take(self.num_sources) {
            let eigenvec = eigenvectors
                .slice_with::<1>(&[SliceArg::All, SliceArg::Index(idx as isize)])
                .expect("eigenvector column slice");

            for i in 0..n {
                for j in 0..n {
                    p_s[[i, j]] += eigenvec[i] * eigenvec[j].conj();
                }
            }
        }

        let r_inv_a =
            kwavers_math::linear_algebra::ComplexLinearAlgebra::solve_linear_system_complex(
                &r_loaded, steering,
            )?;

        let mut ps_r_inv_a = Array1::<Complex64>::from_elem(n, Complex64::default());
        for i in 0..n {
            for j in 0..n {
                ps_r_inv_a[i] += p_s[[i, j]] * r_inv_a[j];
            }
        }

        let mut a_h_r_inv_ps_a = Complex64::default();
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

        for &w in weights.iter() {
            if !w.re.is_finite() || !w.im.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    "EigenspaceMV::compute_weights: non-finite weight computed".to_owned(),
                )));
            }
        }

        Ok(weights)
    }

    /// Eigenspace signal-subspace projector response `b_ES = |aᴴ P_s a|²`.
    ///
    /// This is the localization map of Theorem 22.2 (passive acoustic mapping):
    /// the steering vector `a(r_f)` is projected onto the signal subspace
    /// `P_s = E_s E_sᴴ` spanned by the `num_sources` largest eigenvectors of `R`.
    /// Because `P_s` is a Hermitian projector, `aᴴ P_s a = ‖P_s a‖² ≥ 0` is real;
    /// the returned value is its square, peaking where `a(r_f)` aligns with the
    /// source subspace and collapsing toward the noise-only directions.
    ///
    /// Unlike [`Self::compute_weights`], no covariance inverse is formed — this is
    /// a pure subspace-projection map, so it does not require `R` to be invertible.
    ///
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` for a non-square/empty `covariance`,
    ///   a steering-length mismatch, or `num_sources >= N`.
    /// - Returns [`KwaversError::Numerical`] for non-finite steering input or output.
    /// - Propagates eigendecomposition failures.
    pub fn signal_subspace_response(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<f64> {
        let n = covariance.shape()[0];

        if n == 0 || covariance.shape()[1] != n {
            return Err(KwaversError::InvalidInput(
                "EigenspaceMV::signal_subspace_response: covariance must be non-empty square matrix"
                    .to_owned(),
            ));
        }
        if steering.len() != n {
            return Err(KwaversError::InvalidInput(format!(
                "EigenspaceMV::signal_subspace_response: steering length {} != covariance dim {n}",
                steering.len()
            )));
        }
        if self.num_sources == 0 || self.num_sources >= n {
            return Err(KwaversError::InvalidInput(format!(
                "EigenspaceMV::signal_subspace_response: num_sources {} must satisfy 0 < K < N {n}",
                self.num_sources
            )));
        }
        for &val in steering.iter() {
            if !val.re.is_finite() || !val.im.is_finite() {
                return Err(KwaversError::Numerical(NumericalError::NaN {
                    operation: "EigenspaceMV::signal_subspace_response".to_owned(),
                    inputs: "steering vector contains non-finite values".to_owned(),
                }));
            }
        }

        let (eigenvalues, eigenvectors) = {
            let r = EigenSolver::jacobi_hermitian(covariance, EigenSolverConfig::default())?;
            (r.eigenvalues, r.eigenvectors)
        };

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));

        // aᴴ P_s a = Σ_{k<K} |e_kᴴ a|²  (P_s = Σ_{k<K} e_k e_kᴴ, Hermitian projector).
        let mut projection_power = 0.0_f64;
        for &idx in indices.iter().take(self.num_sources) {
            let eigenvec = eigenvectors
                .slice_with::<1>(&[SliceArg::All, SliceArg::Index(idx as isize)])
                .expect("eigenvector column slice");
            let mut e_h_a = Complex64::default();
            for j in 0..n {
                e_h_a += eigenvec[j].conj() * steering[j];
            }
            projection_power += e_h_a.norm_sqr();
        }

        let response = projection_power * projection_power; // |aᴴ P_s a|²
        if !response.is_finite() {
            return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                "EigenspaceMV::signal_subspace_response: non-finite response".to_owned(),
            )));
        }
        Ok(response)
    }
}