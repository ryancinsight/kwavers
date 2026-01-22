//! Covariance estimation for beamforming
//!
//! ## Mathematical Foundation
//! **Sample Covariance**: `R = (1/N) Σ x_n x_nᴴ`
//! **Forward-Backward Averaging**: Reduces correlation matrix bias for finite samples
//! **Spatial Smoothing**: Decorrelates coherent sources using subarray averaging
//! **Shrinkage**: Improves conditioning by mixing the sample covariance with a structured target
//!
//! ## Narrowband / complex-baseband note (MVDR/MUSIC correctness)
//! Narrowband adaptive methods (MVDR/Capon, MUSIC, ESMV) are naturally expressed on **complex**
//! snapshots `x_n ∈ ℂ^M` with a **Hermitian** covariance `R = E[x xᴴ]`.
//!
//! This module therefore provides both:
//! - real-valued covariance estimation (`estimate`) for pragmatic baseline processing, and
//! - complex-valued covariance estimation (`estimate_complex`) for mathematically correct narrowband
//!   baseband snapshot processing.
//!
//! # Invariants
//! - `num_sensors > 0`
//! - `num_snapshots > 0`
//! - For `estimate_complex`, the output is Hermitian up to floating-point roundoff.
//!
//! # Strategy enums (advanced / literature-aligned)
//! This file provides explicit enums for covariance post-processing so higher layers can select
//! appropriate conditioning and coherent-source handling without reimplementing numerics.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2};
use num_complex::Complex64;

/// Covariance post-processing strategy.
///
/// This is an explicit, invariant-driven policy enum.
/// It is intentionally separate from diagonal loading (which is applied by MVDR/MUSIC scorers).
#[derive(Debug, Clone, PartialEq)]
pub enum CovariancePostProcess {
    /// No post-processing beyond optional forward-backward averaging (if enabled on the estimator).
    None,
    /// Apply **shrinkage** toward a scaled identity target:
    ///
    /// `R' = (1 - α) R + α μ I`, where `μ = tr(R)/M` and `α ∈ [0,1]`.
    ///
    /// This is a standard conditioning technique and is a common “advanced default” when snapshot
    /// count is low or covariance is near singular.
    ShrinkageToIdentity {
        /// Shrinkage factor `α ∈ [0,1]`.
        alpha: f64,
    },
    /// Apply **spatial smoothing** (subarray averaging) for coherent source decorrelation.
    ///
    /// Notes:
    /// - This reduces aperture (subarray size) and thus resolution, but enables subspace methods
    ///   to handle coherent multipath/coherent tones.
    SpatialSmoothing {
        /// Subarray size (must be in [1, M]).
        subarray_size: usize,
    },
    /// Apply shrinkage then spatial smoothing.
    ///
    /// Ordering is explicit and deterministic.
    ShrinkageThenSpatialSmoothing {
        /// Shrinkage factor `α ∈ [0,1]`.
        alpha: f64,
        /// Subarray size (must be in [1, M]).
        subarray_size: usize,
    },
}

impl Default for CovariancePostProcess {
    fn default() -> Self {
        // Advanced default: slight shrinkage to improve conditioning without noticeably biasing.
        Self::ShrinkageToIdentity { alpha: 0.05 }
    }
}

impl CovariancePostProcess {
    /// Validate invariants of the policy itself.
    pub fn validate(&self) -> KwaversResult<()> {
        match self {
            Self::None => Ok(()),
            Self::ShrinkageToIdentity { alpha } => {
                if !alpha.is_finite() || *alpha < 0.0 || *alpha > 1.0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::ShrinkageToIdentity: alpha must be finite and in [0,1]"
                            .to_string(),
                    ));
                }
                Ok(())
            }
            Self::SpatialSmoothing { subarray_size } => {
                if *subarray_size == 0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::SpatialSmoothing: subarray_size must be >= 1"
                            .to_string(),
                    ));
                }
                Ok(())
            }
            Self::ShrinkageThenSpatialSmoothing {
                alpha,
                subarray_size,
            } => {
                if !alpha.is_finite() || *alpha < 0.0 || *alpha > 1.0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::ShrinkageThenSpatialSmoothing: alpha must be finite and in [0,1]"
                            .to_string(),
                    ));
                }
                if *subarray_size == 0 {
                    return Err(KwaversError::InvalidInput(
                        "CovariancePostProcess::ShrinkageThenSpatialSmoothing: subarray_size must be >= 1"
                            .to_string(),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Covariance matrix estimator with multiple methods
#[derive(Debug, Clone)]
pub struct CovarianceEstimator {
    /// Use forward-backward averaging for improved estimation
    pub forward_backward_averaging: bool,
    /// Number of snapshots for averaging
    pub num_snapshots: usize,
    /// Post-processing strategy (shrinkage / spatial smoothing).
    pub post_process: CovariancePostProcess,
}

impl Default for CovarianceEstimator {
    fn default() -> Self {
        Self {
            forward_backward_averaging: true,
            num_snapshots: 1,
            post_process: CovariancePostProcess::default(),
        }
    }
}

impl CovarianceEstimator {
    /// Create new covariance estimator
    #[must_use]
    pub fn new(forward_backward_averaging: bool, num_snapshots: usize) -> Self {
        Self {
            forward_backward_averaging,
            num_snapshots,
            post_process: CovariancePostProcess::default(),
        }
    }

    /// Create a new covariance estimator with an explicit post-processing policy.
    #[must_use]
    pub fn with_post_process(
        forward_backward_averaging: bool,
        num_snapshots: usize,
        post_process: CovariancePostProcess,
    ) -> Self {
        Self {
            forward_backward_averaging,
            num_snapshots,
            post_process,
        }
    }

    /// Estimate sample covariance matrix (real-valued): `R = (1/N) Σ x_n x_nᵀ`.
    ///
    /// This is a pragmatic baseline used for broad compatibility with real-valued pipelines.
    /// For narrowband MVDR/MUSIC where complex steering is used, prefer `estimate_complex`
    /// on complex baseband snapshots with `R = (1/N) Σ x_n x_nᴴ`.
    pub fn estimate(&self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (num_sensors, num_snapshots) = data.dim();

        if num_sensors == 0 || num_snapshots == 0 {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::estimate: data must have shape (num_sensors>0, num_snapshots>0)"
                    .to_string(),
            ));
        }

        // Initialize covariance matrix
        let mut covariance = Array2::zeros((num_sensors, num_sensors));

        // Compute sample covariance: R = (1/N) Σ x_n x_nᵀ
        for snapshot in 0..num_snapshots {
            let x = data.column(snapshot);
            for i in 0..num_sensors {
                for j in 0..num_sensors {
                    covariance[[i, j]] += x[i] * x[j];
                }
            }
        }

        // Normalize by number of snapshots
        covariance.mapv_inplace(|x| x / num_snapshots as f64);

        // Apply forward-backward averaging if enabled
        if self.forward_backward_averaging {
            covariance = self.apply_forward_backward_averaging(&covariance);
        }

        // Apply configured post-processing
        covariance = self.apply_post_process_real(&covariance)?;

        Ok(covariance)
    }

    /// Estimate sample covariance matrix (complex-valued): `R = (1/N) Σ x_n x_nᴴ`.
    ///
    /// # Intended use
    /// Narrowband adaptive processing (MVDR/Capon spectra, MUSIC pseudospectrum, ESMV) on complex
    /// baseband snapshots.
    ///
    /// # Errors
    /// Returns an error if `data` has zero sensors or zero snapshots.
    pub fn estimate_complex(&self, data: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>> {
        let (num_sensors, num_snapshots) = data.dim();

        if num_sensors == 0 || num_snapshots == 0 {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::estimate_complex: data must have shape (num_sensors>0, num_snapshots>0)"
                    .to_string(),
            ));
        }

        let mut covariance = Array2::<Complex64>::zeros((num_sensors, num_sensors));

        // R = (1/N) Σ x xᴴ
        for snapshot in 0..num_snapshots {
            let x = data.column(snapshot);
            for i in 0..num_sensors {
                for j in 0..num_sensors {
                    covariance[(i, j)] += x[i] * x[j].conj();
                }
            }
        }

        let inv_n = 1.0 / (num_snapshots as f64);
        covariance.mapv_inplace(|v| v * inv_n);

        if self.forward_backward_averaging {
            covariance = self.apply_forward_backward_averaging_complex(&covariance);
        }

        covariance = self.apply_post_process_complex(&covariance)?;

        Ok(covariance)
    }

    /// Apply forward-backward averaging to reduce estimation bias (real-valued).
    ///
    /// `R_fb = 0.5 * (R + J R* J)` where `J` is the exchange matrix.
    #[must_use]
    pub fn apply_forward_backward_averaging(&self, covariance: &Array2<f64>) -> Array2<f64> {
        let n = covariance.nrows();
        let mut fb_covariance = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let forward = covariance[[i, j]];
                let backward = covariance[[n - 1 - i, n - 1 - j]]; // J R* J for real matrices
                fb_covariance[[i, j]] = 0.5 * (forward + backward);
            }
        }

        fb_covariance
    }

    /// Apply forward-backward averaging to reduce estimation bias (complex-valued).
    ///
    /// `R_fb = 0.5 * (R + J R* J)` where `R*` is elementwise complex conjugate and `J` reverses
    /// row/column order (exchange matrix).
    #[must_use]
    pub fn apply_forward_backward_averaging_complex(
        &self,
        covariance: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        let n = covariance.nrows();
        let mut fb = Array2::<Complex64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let forward = covariance[(i, j)];
                // (J R* J)_{i,j} = conj( R_{n-1-i, n-1-j} )
                let backward = covariance[(n - 1 - i, n - 1 - j)].conj();
                fb[(i, j)] = (forward + backward) * 0.5;
            }
        }

        fb
    }

    fn apply_post_process_real(&self, covariance: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        self.post_process.validate()?;
        let m = covariance.nrows();
        if m == 0 || covariance.ncols() != m {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::apply_post_process_real: covariance must be square and non-empty"
                    .to_string(),
            ));
        }

        match &self.post_process {
            CovariancePostProcess::None => Ok(covariance.clone()),

            CovariancePostProcess::ShrinkageToIdentity { alpha } => {
                Ok(shrinkage_to_identity_real(covariance, *alpha))
            }

            CovariancePostProcess::SpatialSmoothing { subarray_size } => {
                let smoother = SpatialSmoothing::new(*subarray_size);
                smoother.apply(covariance)
            }

            CovariancePostProcess::ShrinkageThenSpatialSmoothing {
                alpha,
                subarray_size,
            } => {
                let shrunk = shrinkage_to_identity_real(covariance, *alpha);
                let smoother = SpatialSmoothing::new(*subarray_size);
                smoother.apply(&shrunk)
            }
        }
    }

    fn apply_post_process_complex(
        &self,
        covariance: &Array2<Complex64>,
    ) -> KwaversResult<Array2<Complex64>> {
        self.post_process.validate()?;
        let m = covariance.nrows();
        if m == 0 || covariance.ncols() != m {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::apply_post_process_complex: covariance must be square and non-empty"
                    .to_string(),
            ));
        }

        match &self.post_process {
            CovariancePostProcess::None => Ok(covariance.clone()),

            CovariancePostProcess::ShrinkageToIdentity { alpha } => {
                Ok(shrinkage_to_identity_complex(covariance, *alpha))
            }

            CovariancePostProcess::SpatialSmoothing { subarray_size } => {
                let smoother = SpatialSmoothingComplex::new(*subarray_size);
                smoother.apply(covariance)
            }

            CovariancePostProcess::ShrinkageThenSpatialSmoothing {
                alpha,
                subarray_size,
            } => {
                let shrunk = shrinkage_to_identity_complex(covariance, *alpha);
                let smoother = SpatialSmoothingComplex::new(*subarray_size);
                smoother.apply(&shrunk)
            }
        }
    }

    /// Estimate covariance matrix with spatial smoothing for coherent sources (real-valued only).
    pub fn estimate_with_spatial_smoothing(
        &self,
        data: &Array2<f64>,
        subarray_size: usize,
    ) -> KwaversResult<Array2<f64>> {
        let base_covariance = self.estimate(data)?;
        let spatial_smoothing = SpatialSmoothing::new(subarray_size);
        spatial_smoothing.apply(&base_covariance)
    }

    /// Estimate covariance matrix with spatial smoothing for coherent sources (complex-valued).
    pub fn estimate_complex_with_spatial_smoothing(
        &self,
        data: &Array2<Complex64>,
        subarray_size: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        let base_covariance = self.estimate_complex(data)?;
        let spatial_smoothing = SpatialSmoothingComplex::new(subarray_size);
        spatial_smoothing.apply(&base_covariance)
    }
}

fn shrinkage_to_identity_real(covariance: &Array2<f64>, alpha: f64) -> Array2<f64> {
    let m = covariance.nrows().max(1);
    let mut out = covariance.clone();

    // μ = tr(R)/M
    let mut trace = 0.0;
    for i in 0..m {
        trace += covariance[(i, i)];
    }
    let mu = trace / (m as f64);

    // R' = (1-α)R + α μ I
    out.mapv_inplace(|v| (1.0 - alpha) * v);
    for i in 0..m {
        out[(i, i)] += alpha * mu;
    }
    out
}

fn shrinkage_to_identity_complex(covariance: &Array2<Complex64>, alpha: f64) -> Array2<Complex64> {
    let m = covariance.nrows().max(1);
    let mut out = covariance.clone();

    // μ = tr(R)/M is real for Hermitian R; we use the real part explicitly.
    let mut trace_re = 0.0;
    for i in 0..m {
        trace_re += covariance[(i, i)].re;
    }
    let mu = trace_re / (m as f64);

    out.mapv_inplace(|v| v * (1.0 - alpha));
    for i in 0..m {
        out[(i, i)] += Complex64::new(alpha * mu, 0.0);
    }
    out
}

/// Spatial smoothing for coherent source decorrelation (real-valued)
#[derive(Debug, Clone)]
pub struct SpatialSmoothing {
    /// Size of subarrays for smoothing
    pub subarray_size: usize,
}

impl SpatialSmoothing {
    /// Create new spatial smoothing processor
    #[must_use]
    pub fn new(subarray_size: usize) -> Self {
        Self { subarray_size }
    }

    /// Apply spatial smoothing to covariance matrix
    /// Creates multiple subarray covariance matrices and averages them
    pub fn apply(&self, covariance: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = covariance.nrows();

        if self.subarray_size >= n {
            return Ok(covariance.clone());
        }

        let num_subarrays = n - self.subarray_size + 1;
        let mut smoothed = Array2::zeros((self.subarray_size, self.subarray_size));

        // Average covariance matrices from all possible subarrays
        for start_idx in 0..num_subarrays {
            let end_idx = start_idx + self.subarray_size;

            // Extract subarray covariance matrix
            let sub_cov = covariance.slice(s![start_idx..end_idx, start_idx..end_idx]);

            // Add to smoothed matrix
            smoothed += &sub_cov;
        }

        // Normalize by number of subarrays
        smoothed.mapv_inplace(|x| x / num_subarrays as f64);

        Ok(smoothed)
    }
}

/// Spatial smoothing for coherent source decorrelation (complex-valued).
#[derive(Debug, Clone)]
pub struct SpatialSmoothingComplex {
    /// Size of subarrays for smoothing
    pub subarray_size: usize,
}

impl SpatialSmoothingComplex {
    /// Create new spatial smoothing processor
    #[must_use]
    pub fn new(subarray_size: usize) -> Self {
        Self { subarray_size }
    }

    /// Apply spatial smoothing to a complex covariance matrix.
    ///
    /// For narrowband spaces, spatial smoothing is performed by averaging principal submatrices:
    /// `R_smooth = (1/L) ∑ R[start..start+p, start..start+p]`.
    pub fn apply(&self, covariance: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>> {
        let n = covariance.nrows();
        if covariance.ncols() != n {
            return Err(KwaversError::InvalidInput(
                "SpatialSmoothingComplex::apply: covariance must be square".to_string(),
            ));
        }

        if self.subarray_size >= n {
            return Ok(covariance.clone());
        }
        if self.subarray_size == 0 {
            return Err(KwaversError::InvalidInput(
                "SpatialSmoothingComplex::apply: subarray_size must be >= 1".to_string(),
            ));
        }

        let num_subarrays = n - self.subarray_size + 1;
        let mut smoothed = Array2::<Complex64>::zeros((self.subarray_size, self.subarray_size));

        for start_idx in 0..num_subarrays {
            let end_idx = start_idx + self.subarray_size;
            let sub_cov = covariance.slice(s![start_idx..end_idx, start_idx..end_idx]);
            smoothed += &sub_cov;
        }

        let inv = 1.0 / (num_subarrays as f64);
        smoothed.mapv_inplace(|v| v * inv);

        Ok(smoothed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn estimate_complex_is_hermitian_for_simple_data() {
        let mut x = Array2::<Complex64>::zeros((2, 3));
        x[(0, 0)] = Complex64::new(1.0, 2.0);
        x[(1, 0)] = Complex64::new(-0.5, 0.25);
        x[(0, 1)] = Complex64::new(0.1, -0.2);
        x[(1, 1)] = Complex64::new(0.3, 0.4);
        x[(0, 2)] = Complex64::new(-1.0, 0.0);
        x[(1, 2)] = Complex64::new(0.0, 1.0);

        let est = CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        };

        let r = est.estimate_complex(&x).expect("covariance");

        // Hermitian check: R_ij = conj(R_ji)
        for i in 0..2 {
            for j in 0..2 {
                let lhs = r[(i, j)];
                let rhs = r[(j, i)].conj();
                assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-12);
                assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn estimate_complex_rejects_empty() {
        let x = Array2::<Complex64>::zeros((0, 0));
        let est = CovarianceEstimator::default();
        let err = est.estimate_complex(&x).expect_err("must reject empty");
        assert!(err.to_string().contains("estimate_complex"));
    }

    #[test]
    fn forward_backward_averaging_complex_preserves_hermitian_structure() {
        // Start with a Hermitian matrix and verify the transform keeps it Hermitian.
        let mut r = Array2::<Complex64>::zeros((3, 3));
        r[(0, 0)] = Complex64::new(2.0, 0.0);
        r[(1, 1)] = Complex64::new(3.0, 0.0);
        r[(2, 2)] = Complex64::new(4.0, 0.0);
        r[(0, 1)] = Complex64::new(0.5, 0.25);
        r[(1, 0)] = r[(0, 1)].conj();
        r[(1, 2)] = Complex64::new(-0.2, 0.1);
        r[(2, 1)] = r[(1, 2)].conj();
        r[(0, 2)] = Complex64::new(0.0, -0.3);
        r[(2, 0)] = r[(0, 2)].conj();

        let est = CovarianceEstimator {
            forward_backward_averaging: true,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        };

        let fb = est.apply_forward_backward_averaging_complex(&r);

        for i in 0..3 {
            for j in 0..3 {
                let lhs = fb[(i, j)];
                let rhs = fb[(j, i)].conj();
                assert_abs_diff_eq!(lhs.re, rhs.re, epsilon = 1e-12);
                assert_abs_diff_eq!(lhs.im, rhs.im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn shrinkage_to_identity_real_preserves_symmetry_and_improves_diagonal() {
        let mut r = Array2::<f64>::zeros((2, 2));
        r[(0, 0)] = 1.0;
        r[(1, 1)] = 3.0;
        r[(0, 1)] = 0.2;
        r[(1, 0)] = 0.2;

        let shrunk = shrinkage_to_identity_real(&r, 0.5);

        // Symmetry preserved
        assert_abs_diff_eq!(shrunk[(0, 1)], shrunk[(1, 0)], epsilon = 1e-15);
        // Diagonal remains finite
        assert!(shrunk[(0, 0)].is_finite() && shrunk[(1, 1)].is_finite());
    }

    #[test]
    fn spatial_smoothing_complex_shapes_match() {
        let mut r = Array2::<Complex64>::zeros((4, 4));
        for i in 0..4 {
            r[(i, i)] = Complex64::new(1.0 + i as f64, 0.0);
        }

        let smoother = SpatialSmoothingComplex::new(3);
        let sm = smoother.apply(&r).expect("smoothed");
        assert_eq!(sm.nrows(), 3);
        assert_eq!(sm.ncols(), 3);
    }
}
