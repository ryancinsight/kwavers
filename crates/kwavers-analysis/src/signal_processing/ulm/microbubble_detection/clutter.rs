//! SVD spatiotemporal clutter filter (Demené et al. 2015).
//!
//! # Theorem: Spatiotemporal SVD Clutter Filtering
//!
//! **Theorem** (Demené et al. 2015 §II-B):
//! The IQ data matrix S ∈ ℝ^{N_px × N_t} has the property that tissue clutter
//! concentrates in the first k left singular vectors (correlated over time),
//! while microbubbles decorrelate rapidly (δ-correlated).
//!
//! ```text
//! Decompose:  S = U Σ Vᵀ
//! Tissue:     T = U[:,0:k] diag(Σ[0:k]) Vᵀ[0:k,:]
//! Bubble:     B = S − T   (sparse, transient signal)
//! ```
//!
//! The optimal threshold k is given by the Singular Value Hard Threshold (SVHT)
//! (Gavish & Donoho 2014):
//! ```text
//! τ = ω(β) · σ_median     (the universal threshold)
//! where β = N_px/N_t (aspect ratio)
//! ω(β) = 0.56·β³ − 0.95·β² + 1.82·β + 1.43   (Gavish & Donoho 2014, eq. 11)
//! σ_median = median(σᵢ) / (0.6745 · √(N_t))    (scaled noise estimate)
//! k = max{n : σₙ > τ}
//! ```

use super::types::SvdClutterConfig;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{Array1, Array2, SliceArg};
use leto_ops::svd_decompose;

/// SVD spatiotemporal clutter filter.
///
/// Separates microbubble signal from tissue clutter by projecting out the
/// low-rank tissue subspace in the IQ data matrix.
#[derive(Debug)]
pub struct UlmSvdClutterFilter {
    config: SvdClutterConfig,
}

impl UlmSvdClutterFilter {
    #[must_use]
    pub fn new(config: SvdClutterConfig) -> Self {
        Self { config }
    }

    /// Filter IQ data matrix S \[N_px × N_t\] → bubble-only matrix B \[N_px × N_t\].
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. [U, Σ, Vᵀ] = SVD(S)
    /// 2. τ = SVHT(β, Σ);  k = max{n : σₙ > τ} + rank_margin
    /// 3. T = U[:,0:k] · diag(Σ[0:k]) · Vᵀ[0:k,:]
    /// 4. B = S − T
    /// ```
    ///
    /// Returns `(B, k)` where k is the tissue rank used.
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn filter(&self, iq_data: &Array2<f64>) -> KwaversResult<(Array2<f64>, usize)> {
        let [n_px, n_t] = iq_data.shape();
        if n_px == 0 || n_t == 0 {
            return Err(KwaversError::Numerical(NumericalError::SolverFailed {
                method: "SVD clutter filter".to_owned(),
                reason: "empty IQ data matrix".to_owned(),
            }));
        }

        let svd = svd_decompose(&iq_data.view())?;
        let u = svd.left_singular_vectors;
        let vt = svd.right_singular_vectors;
        let sigma = Array1::from_vec(svd.singular_values.len(), svd.singular_values)?;

        let k = if self.config.fixed_clutter_rank > 0 {
            self.config.fixed_clutter_rank.min(sigma.len())
        } else {
            let k_svht = svht_threshold(&sigma, n_px, n_t);
            (k_svht + self.config.rank_margin).min(sigma.len())
        };

        if k == 0 {
            return Ok((iq_data.clone(), 0));
        }

        // Reconstruct tissue: T = U[:,0:k] · diag(Σ[0:k]) · Vᵀ[0:k,:]
        //
        // LinearAlgebra::svd returns (U, Σ, V) where V columns = right singular vectors.
        // Reconstruction: V_k^T = first k columns of V, transposed.
        let tissue = {
            let u_k = u
                .slice_with::<2>(&[
                    SliceArg::All,
                    SliceArg::Range { start: Some(0), end: Some(k as isize), step: 1 },
                ])
                .expect("invariant: SVD U column slice within bounds")
                .to_contiguous();
            let sigma_k = sigma
                .slice_with::<1>(&[SliceArg::Range {
                    start: Some(0),
                    end: Some(k as isize),
                    step: 1,
                }])
                .expect("invariant: sigma slice within bounds")
                .to_contiguous();
            let v_k = vt
                .slice_with::<2>(&[
                    SliceArg::All,
                    SliceArg::Range { start: Some(0), end: Some(k as isize), step: 1 },
                ])
                .expect("invariant: SVD V column slice within bounds")
                .to_contiguous();
            let vt_k = v_k
                .transpose([1, 0])
                .expect("invariant: 2-D transpose axes valid");
            let mut us = u_k;
            for (j, &s_j) in sigma_k.iter().enumerate() {
                for i in 0..n_px {
                    us[[i, j]] *= s_j;
                }
            }
            let mut result = Array2::<f64>::zeros((n_px, n_t));
            leto_ops::matmul(&us.view(), &vt_k, &mut result.view_mut())
                .expect("invariant: tissue reconstruction dimensions match");
            result
        };

        let bubble = iq_data - &tissue;
        Ok((bubble, k))
    }
}

/// Singular Value Hard Threshold (Gavish & Donoho 2014).
///
/// # Theorem (Gavish & Donoho 2014, Theorem 3 + Supplementary S5)
///
/// For an n×m matrix (n ≤ m) corrupted by iid Gaussian noise with unknown variance σ²,
/// β = n/m, the optimal hard threshold (unknown-σ case) is:
/// ```text
/// τ = ω(β) · s_med / μ_MP(β)
/// ```
/// where:
/// - `ω(β) = 0.56·β³ − 0.95·β² + 1.82·β + 1.43`  (SVHT coefficient, eq. 11)
/// - `s_med` = median singular value
/// - `μ_MP(β)` = theoretical median of the Marchenko-Pastur distribution with parameter β
///
/// Returns the number of singular values that exceed the threshold (tissue rank k).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[must_use]
pub(super) fn svht_threshold(sigma: &Array1<f64>, n_rows: usize, n_cols: usize) -> usize {
    let (n, m) = if n_rows <= n_cols {
        (n_rows, n_cols)
    } else {
        (n_cols, n_rows)
    };
    let beta = n as f64 / m as f64;

    // ω(β) polynomial approximation (Gavish & Donoho 2014, eq. 11)
    let omega = 1.82f64.mul_add(beta, 0.56f64.mul_add(beta.powi(3), -(0.95 * beta.powi(2)))) + 1.43;

    let mut s_sorted: Vec<f64> = sigma.iter().copied().collect();
    s_sorted.sort_by(|a, b| a.total_cmp(b));
    let s_med = if s_sorted.is_empty() {
        return 0;
    } else if s_sorted.len().is_multiple_of(2) {
        (s_sorted[s_sorted.len() / 2 - 1] + s_sorted[s_sorted.len() / 2]) / 2.0
    } else {
        s_sorted[s_sorted.len() / 2]
    };

    let mu_mp = mp_median(beta);
    let tau = omega * s_med / mu_mp;

    sigma.iter().filter(|&&sv| sv > tau).count()
}

/// Numerical median of the Marchenko-Pastur distribution with parameter β ∈ (0, 1].
///
/// # Theorem
///
/// The Marchenko-Pastur distribution with parameter β ∈ (0,1] has support
/// [λ₋, λ₊] = [(1−√β)², (1+√β)²] and PDF:
/// ```text
/// f(x) = √((λ₊−x)(x−λ₋)) / (2π·β·x)
/// ```
///
/// # Algorithm
///
/// Uses substitution `x = λ₋ + (λ₊−λ₋)·sin²(φ)` to avoid endpoint singularities,
/// then bisects on the CDF to find the median.
fn mp_median(beta: f64) -> f64 {
    let lb = (1.0_f64 - beta.sqrt()).powi(2);
    let ub = (1.0_f64 + beta.sqrt()).powi(2);
    let range = ub - lb;

    let cdf = |x: f64| -> f64 {
        if x <= lb {
            return 0.0;
        }
        if x >= ub {
            return 1.0;
        }
        let phi_max = ((x - lb) / range).sqrt().asin();
        let n = 400usize;
        let dphi = phi_max / n as f64;
        let mut sum = 0.0;
        for k in 0..n {
            let phi = (k as f64 + 0.5) * dphi;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();
            let xv = (range * sin_phi).mul_add(sin_phi, lb);
            let sin2 = 2.0 * sin_phi * cos_phi;
            let integrand = range * range * sin2 * sin2 / (TWO_PI * beta * xv);
            sum += integrand * dphi;
        }
        sum
    };

    let cdf_total = cdf(ub - 1e-10);

    let mut lo = lb;
    let mut hi = ub;
    for _ in 0..52 {
        let mid = (lo + hi) / 2.0;
        if cdf(mid) / cdf_total < 0.5 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}
