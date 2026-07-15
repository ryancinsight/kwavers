//! Penalized weighted least squares (PWLS) data weighting — the low-dose-CT lesson.
//!
//! Model-based iterative reconstruction (MBIR) for **low-dose CT** does not
//! minimise an unweighted least-squares data misfit. It minimises a *statistically
//! weighted* one,
//! ```text
//!   J(x) = ½ (y − Ax)ᵀ W (y − Ax) + β R(x),   W = diag(1/σ_i²),
//! ```
//! where `σ_i²` is the noise variance of measurement `i` (Sauer & Bouman 1993;
//! Thibault et al. 2007; Fessler). In CT the per-ray variance follows from photon
//! statistics, `σ_i² ≈ exp(p_i)/I₀`, so low-count rays through dense bone are
//! **down-weighted** instead of corrupting the reconstruction with equal authority.
//!
//! The same statistics-aware data fidelity transfers directly to transcranial
//! ultrasound FWI: traces that traverse the skull are strongly attenuated and
//! noise-dominated, yet the default unweighted L2 misfit
//! (`reconstruction::seismic::misfit::norm_metrics`) lets every trace contribute
//! to the gradient with equal weight. Weighting each trace by its inverse noise
//! variance is the maximum-likelihood estimator under heteroscedastic Gaussian
//! noise (the Gauss–Markov / BLUE result) and is the FWI analogue of CT's PWLS.
//!
//! Unlike CT there is no closed-form photon-count variance; the noise level is
//! instead estimated per trace from a quiet **pre-first-arrival window** (the
//! leading samples that contain only measurement noise), the standard SNR
//! estimator in seismic/ultrasound practice.

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::Array2;

/// Data-fidelity weighting strategy for the L2 misfit (the PWLS / MBIR lesson).
///
/// `Uniform` reproduces the classical unweighted least-squares misfit exactly;
/// `InverseNoiseVariance` reproduces the low-dose-CT PWLS estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataWeighting {
    /// Plain L2: every trace and sample contributes with equal weight.
    #[default]
    Uniform,
    /// PWLS: weight each trace by its inverse noise variance `1/σ_r²`, with the
    /// variance estimated from the first `noise_window` (pre-first-arrival)
    /// samples of that trace. Weights are mean-normalised to 1 so the objective
    /// scale — and hence a tuned step size — is preserved relative to plain L2.
    InverseNoiseVariance {
        /// Number of leading samples used to estimate each trace's noise variance.
        noise_window: usize,
    },
}

/// Fraction of the median noise variance used as a floor, so a single near-silent
/// trace cannot acquire unbounded weight (`σ² → 0 ⇒ w → ∞`).
const VARIANCE_FLOOR_FRACTION: f64 = 1.0e-3;

/// Per-trace PWLS weights `w_r`, broadcast across time, mean-normalised to 1.
///
/// Each row of `observed` is one trace `(n_traces, n_time)`. For
/// [`DataWeighting::Uniform`] this is an all-ones array (so the weighted misfit
/// reduces to plain L2 bit-for-bit). For [`DataWeighting::InverseNoiseVariance`]
/// the weight of trace `r` is `1/max(σ_r², floor)` rescaled so the mean weight is
/// 1, where `σ_r²` is the sample variance of the leading `noise_window` samples.
#[must_use]
pub fn trace_weights(observed: &Array2<f64>, weighting: DataWeighting) -> Array2<f64> {
    let [n_tr, nt] = observed.shape();
    let noise_window = match weighting {
        DataWeighting::Uniform => return Array2::ones((n_tr, nt)),
        DataWeighting::InverseNoiseVariance { noise_window } => noise_window.clamp(1, nt.max(1)),
    };
    if n_tr == 0 || nt == 0 {
        return Array2::ones((n_tr, nt));
    }

    // Per-trace noise variance from the leading (signal-free) window.
    let mut var = vec![0.0_f64; n_tr];
    for r in 0..n_tr {
        let mut mean = 0.0;
        for t in 0..noise_window {
            mean += observed[[r, t]];
        }
        mean /= noise_window as f64;
        let mut s = 0.0;
        for t in 0..noise_window {
            let d = observed[[r, t]] - mean;
            s += d * d;
        }
        var[r] = s / noise_window as f64;
    }

    // Floor the variance at a small fraction of the median positive variance.
    let mut positive: Vec<f64> = var.iter().copied().filter(|v| *v > 0.0).collect();
    let floor = if positive.is_empty() {
        1.0
    } else {
        positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        VARIANCE_FLOOR_FRACTION * positive[(positive.len()) / 2]
    };

    let inv: Vec<f64> = var.iter().map(|v| 1.0 / v.max(floor)).collect();
    let mean_w = inv.iter().sum::<f64>() / n_tr as f64;
    let scale = if mean_w > 0.0 { 1.0 / mean_w } else { 1.0 };

    let mut out = Array2::zeros((n_tr, nt));
    for r in 0..n_tr {
        let wr = inv[r] * scale;
        for t in 0..nt {
            out[[r, t]] = wr;
        }
    }
    out
}

fn validate_triple(
    observed: &Array2<f64>,
    synthetic: &Array2<f64>,
    weights: &Array2<f64>,
) -> KwaversResult<()> {
    if observed.shape() != synthetic.shape() || observed.shape() != weights.shape() {
        return Err(KwaversError::Validation(
            ValidationError::ConstraintViolation {
                message: format!(
                    "PWLS shape mismatch: observed {:?}, synthetic {:?}, weights {:?}",
                    observed.shape(),
                    synthetic.shape(),
                    weights.shape()
                ),
            },
        ));
    }
    Ok(())
}

/// Weighted L2 objective `J = (dt/2) Σ_{r,t} w_{r,t} (d_syn − d_obs)²`.
///
/// With all-ones `weights` this equals the unweighted `l2_objective` exactly.
/// # Errors
/// Returns [`crate::KwaversError::Validation`] if the three arrays differ in shape.
pub fn weighted_l2_objective(
    dt: f64,
    observed: &Array2<f64>,
    synthetic: &Array2<f64>,
    weights: &Array2<f64>,
) -> KwaversResult<f64> {
    validate_triple(observed, synthetic, weights)?;
    let mut acc = 0.0;
    for ((&o, &s), &w) in observed.iter().zip(synthetic.iter()).zip(weights.iter()) {
        let r = s - o;
        acc += w * r * r;
    }
    Ok(0.5 * dt * acc)
}

/// Weighted L2 adjoint source `w ⊙ (d_syn − d_obs)` — the gradient of
/// [`weighted_l2_objective`] with respect to the synthetic data.
///
/// With all-ones `weights` this equals the unweighted residual `d_syn − d_obs`.
/// # Errors
/// Returns [`crate::KwaversError::Validation`] if the three arrays differ in shape.
pub fn weighted_l2_residual(
    observed: &Array2<f64>,
    synthetic: &Array2<f64>,
    weights: &Array2<f64>,
) -> KwaversResult<Array2<f64>> {
    validate_triple(observed, synthetic, weights)?;
    Ok(&(synthetic - observed) * weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array2;

    /// Uniform weighting reproduces plain L2 bit-for-bit: all-ones weights, the
    /// weighted objective equals `(dt/2)‖r‖²`, and the weighted residual equals
    /// the plain difference.
    #[test]
    fn uniform_weighting_matches_plain_l2() {
        let obs = Array2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.0, 4.0]).unwrap();
        let syn = Array2::from_shape_vec([2, 3], vec![1.5, 2.0, 2.0, -1.0, 1.0, 4.5]).unwrap();
        let w = trace_weights(&obs, DataWeighting::Uniform);
        assert!(w.iter().all(|&x| (x - 1.0).abs() < 1e-15));

        let dt = 0.5;
        let j_w = weighted_l2_objective(dt, &obs, &syn, &w).unwrap();
        let plain: f64 = (&syn - &obs).mapv(|x| x * x).iter().sum::<f64>();
        assert!(
            (j_w - 0.5 * dt * plain).abs() < 1e-12,
            "weighted == plain L2"
        );

        let r_w = weighted_l2_residual(&obs, &syn, &w).unwrap();
        assert_eq!(r_w, &syn - &obs, "uniform residual == plain difference");
    }

    /// A trace with a large pre-arrival noise window is down-weighted relative to a
    /// quiet one, and the weights are mean-normalised to 1.
    #[test]
    fn noisy_trace_is_downweighted() {
        // Trace 0: quiet pre-window (small noise). Trace 1: loud pre-window (noise).
        let (n_tr, nt, win) = (2usize, 8usize, 4usize);
        let mut obs = Array2::zeros((n_tr, nt));
        // deterministic alternating signs scaled per trace → variance ∝ amplitude².
        for t in 0..nt {
            let sign = if t % 2 == 0 { 1.0 } else { -1.0 };
            obs[[0, t]] = 0.1 * sign; // σ₀ ≈ 0.1
            obs[[1, t]] = 1.0 * sign; // σ₁ ≈ 1.0  (100× the variance)
        }
        let w = trace_weights(
            &obs,
            DataWeighting::InverseNoiseVariance { noise_window: win },
        );
        let w0 = w[[0, 0]];
        let w1 = w[[1, 0]];
        assert!(w0 > w1, "quiet trace up-weighted: w0={w0:.4} w1={w1:.4}");
        // variance ratio ≈ 100 ⇒ weight ratio ≈ 100.
        assert!(
            (w0 / w1 - 100.0).abs() / 100.0 < 0.05,
            "weight ratio tracks 1/σ²"
        );
        let mean_w = {
            let col = w.index_axis::<1>(1, 0).unwrap();
            let (sum, count) = col
                .iter()
                .fold((0.0_f64, 0usize), |(s, c), &v| (s + v, c + 1));
            sum / count as f64
        };
        assert!(
            (mean_w - 1.0).abs() < 1e-9,
            "weights mean-normalised to 1; got {mean_w}"
        );
    }

    /// **Maximum-likelihood property** (the actual low-dose-CT lesson): under
    /// heteroscedastic Gaussian noise, the PWLS (inverse-variance-weighted)
    /// estimate of a common scalar offset is closer to the truth than the
    /// equal-weight mean, averaged over many noise realisations.
    ///
    /// Closed form: minimising `Σ_r w_r (a − y_r)²` gives `a* = Σ w_r y_r / Σ w_r`.
    /// With `w_r = 1/σ_r²` this is the BLUE; with `w_r = 1` it is the plain mean.
    #[test]
    fn pwls_estimator_beats_unweighted_under_heteroscedastic_noise() {
        // Two "traces" measuring the same truth = 0 with very different noise:
        // σ₀ = 0.1 (clean), σ₁ = 1.5 (noisy). PWLS should trust trace 0.
        let (sigma0, sigma1) = (0.1_f64, 1.5_f64);
        let n_real = 4000usize;
        // Box–Muller from a deterministic LCG (reproducible, no Date/rand).
        let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut unif = || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((lcg >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
        };
        let mut gauss = || {
            let u1 = unif().max(1e-12);
            let u2 = unif();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };

        // PWLS weights from the true variances (mean-normalisation cancels in a*).
        let (w0, w1) = (1.0 / (sigma0 * sigma0), 1.0 / (sigma1 * sigma1));
        let (mut err_pwls, mut err_mean) = (0.0_f64, 0.0_f64);
        for _ in 0..n_real {
            let y0 = sigma0 * gauss();
            let y1 = sigma1 * gauss();
            let a_pwls = (w0 * y0 + w1 * y1) / (w0 + w1);
            let a_mean = 0.5 * (y0 + y1);
            err_pwls += a_pwls * a_pwls;
            err_mean += a_mean * a_mean;
        }
        err_pwls /= n_real as f64;
        err_mean /= n_real as f64;
        // Theoretical MSEs: PWLS = 1/(w0+w1) ≈ 0.00998; mean = (σ₀²+σ₁²)/4 ≈ 0.5650.
        let mse_pwls_theory = 1.0 / (w0 + w1);
        let mse_mean_theory = 0.25 * (sigma0 * sigma0 + sigma1 * sigma1);
        assert!(
            err_pwls < 0.25 * err_mean,
            "PWLS MSE {err_pwls:.5} must beat unweighted {err_mean:.5}"
        );
        assert!(
            (err_pwls - mse_pwls_theory).abs() / mse_pwls_theory < 0.1,
            "empirical PWLS MSE {err_pwls:.5} ≈ theory {mse_pwls_theory:.5}"
        );
        assert!(
            (err_mean - mse_mean_theory).abs() / mse_mean_theory < 0.1,
            "empirical mean MSE {err_mean:.5} ≈ theory {mse_mean_theory:.5}"
        );
    }

    /// Shape mismatch is rejected (no silent broadcast).
    #[test]
    fn shape_mismatch_errors() {
        let obs = Array2::<f64>::zeros((2, 3));
        let syn = Array2::<f64>::zeros((2, 3));
        let bad = Array2::<f64>::ones((2, 2));
        assert!(weighted_l2_objective(1.0, &obs, &syn, &bad).is_err());
        assert!(weighted_l2_residual(&obs, &syn, &bad).is_err());
    }
}
