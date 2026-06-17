//! Regularization-parameter selection for Tikhonov-type inverse problems.
//!
//! Given a family of regularized solutions `m_λ` indexed by the regularization
//! weight `λ`, these routines pick the `λ` that best trades data fidelity
//! against model complexity — the L-curve corner (Hansen 1992) and the Morozov
//! discrepancy principle. Book chapter 18 §18.7.
//!
//! Both take the precomputed sampled curves (one `λ` sweep of a real solver);
//! they perform no inversion themselves, so they are reusable across FWI, SIRT,
//! and deconvolution.

/// Index of the **L-curve corner**: the point of maximum curvature of the
/// log–log trade-off curve `(ρ, η) = (ln‖F(m_λ)−d‖, ln‖m_λ‖)` parameterised by
/// `t = ln λ` (Hansen 1992):
///
/// ```text
/// κ(t) = (η′ ρ″ − η″ ρ′) / ((η′² + ρ′²)^{3/2}).
/// ```
///
/// `residual_norms`, `model_norms`, and `lambdas` are sampled over the `λ`
/// sweep (same length `n ≥ 5`, all strictly positive, `lambdas` strictly
/// increasing). Returns the index of the maximum-curvature interior sample, or
/// `None` if the inputs are too short, mismatched, or non-positive.
#[must_use]
pub fn l_curve_corner(
    residual_norms: &[f64],
    model_norms: &[f64],
    lambdas: &[f64],
) -> Option<usize> {
    let n = lambdas.len();
    if n < 5 || residual_norms.len() != n || model_norms.len() != n {
        return None;
    }
    if residual_norms
        .iter()
        .chain(model_norms)
        .chain(lambdas)
        .any(|&v| v.is_nan() || v <= 0.0)
    {
        return None;
    }
    let rho: Vec<f64> = residual_norms.iter().map(|&r| r.ln()).collect();
    let eta: Vec<f64> = model_norms.iter().map(|&m| m.ln()).collect();
    let t: Vec<f64> = lambdas.iter().map(|&l| l.ln()).collect();

    let mut best_idx = None;
    let mut best_kappa = f64::NEG_INFINITY;
    for i in 1..n - 1 {
        let dt_back = t[i] - t[i - 1];
        let dt_fwd = t[i + 1] - t[i];
        if dt_back <= 0.0 || dt_fwd <= 0.0 {
            return None; // lambdas not strictly increasing
        }
        let dt_cen = t[i + 1] - t[i - 1];
        // Central first derivatives w.r.t. t = ln λ.
        let rho_p = (rho[i + 1] - rho[i - 1]) / dt_cen;
        let eta_p = (eta[i + 1] - eta[i - 1]) / dt_cen;
        // Non-uniform central second derivatives.
        let rho_pp = 2.0 * ((rho[i + 1] - rho[i]) / dt_fwd - (rho[i] - rho[i - 1]) / dt_back)
            / dt_cen;
        let eta_pp = 2.0 * ((eta[i + 1] - eta[i]) / dt_fwd - (eta[i] - eta[i - 1]) / dt_back)
            / dt_cen;
        let denom = (eta_p * eta_p + rho_p * rho_p).powf(1.5);
        if denom <= 0.0 {
            continue;
        }
        // The corner is the point of maximum curvature magnitude (the sharpest
        // bend); using |κ| is robust to the sign convention of the formula.
        let kappa = ((eta_p * rho_pp - eta_pp * rho_p) / denom).abs();
        if kappa > best_kappa {
            best_kappa = kappa;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// **Morozov discrepancy principle**: the regularization weight `λ*` at which
/// the data-residual norm equals `τ·δ`, where `δ` is the noise level and
/// `τ ≥ 1` a safety factor. Residual norms are assumed monotone non-decreasing
/// in `λ` (more regularization ⇒ worse data fit); `λ*` is found by linear
/// interpolation between the bracketing samples.
///
/// Returns `None` if `τ·δ` lies outside the sampled residual range, or the
/// inputs are mismatched/empty/non-positive.
#[must_use]
pub fn morozov_lambda(
    lambdas: &[f64],
    residual_norms: &[f64],
    noise_level: f64,
    tau: f64,
) -> Option<f64> {
    let n = lambdas.len();
    if n < 2
        || residual_norms.len() != n
        || noise_level.is_nan()
        || noise_level <= 0.0
        || tau.is_nan()
        || tau < 1.0
    {
        return None;
    }
    let target = tau * noise_level;
    if target < residual_norms[0] || target > residual_norms[n - 1] {
        return None;
    }
    for i in 1..n {
        let (r0, r1) = (residual_norms[i - 1], residual_norms[i]);
        if r1 < r0 {
            return None; // not monotone non-decreasing
        }
        if target <= r1 {
            if (r1 - r0).abs() < f64::MIN_POSITIVE {
                return Some(lambdas[i - 1]);
            }
            let frac = (target - r0) / (r1 - r0);
            return Some(lambdas[i - 1] + frac * (lambdas[i] - lambdas[i - 1]));
        }
    }
    Some(lambdas[n - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l_curve_finds_the_corner_of_an_l_shaped_trade_off() {
        // Logarithmically spaced λ; a Tikhonov-like trade-off whose log–log
        // curve has a sharp corner at index 4: residual stays tiny then climbs,
        // model norm stays large then collapses.
        let lambdas: Vec<f64> = (0..9).map(|i| 10f64.powi(i - 4)).collect();
        let residual = [1.0, 1.02, 1.05, 1.1, 1.3, 3.0, 9.0, 27.0, 81.0];
        let model = [81.0, 27.0, 9.0, 3.0, 1.3, 1.1, 1.05, 1.02, 1.0];
        let corner = l_curve_corner(&residual, &model, &lambdas).expect("corner");
        // The corner sits in the bend region (where both curves turn).
        assert!((3..=5).contains(&corner), "corner index {corner} not at the bend");
    }

    #[test]
    fn l_curve_rejects_degenerate_input() {
        assert_eq!(l_curve_corner(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0]), None);
        let l = [1e-3, 1e-2, 1e-1, 1.0, 10.0];
        assert_eq!(l_curve_corner(&[1.0, 1.0, 1.0, 1.0, -1.0], &[1.0; 5], &l), None);
    }

    #[test]
    fn morozov_interpolates_to_the_target_residual() {
        // Residuals increase with λ; noise δ=2.0, τ=1.0 ⇒ target=2.0 lies
        // between samples 2 (1.5) and 3 (2.5) ⇒ λ* halfway in λ.
        let lambdas = [0.1, 0.2, 0.4, 0.8, 1.6];
        let residual = [0.5, 1.0, 1.5, 2.5, 4.0];
        let lam = morozov_lambda(&lambdas, &residual, 2.0, 1.0).expect("λ*");
        // target 2.0 is 50% from 1.5→2.5, between λ=0.4 and 0.8 ⇒ 0.6.
        assert!((lam - 0.6).abs() < 1e-12, "λ* = {lam}");
        // A residual achieved exactly at a sample returns that λ.
        let lam2 = morozov_lambda(&lambdas, &residual, 1.5, 1.0).expect("λ*");
        assert!((lam2 - 0.4).abs() < 1e-12, "λ* = {lam2}");
    }

    #[test]
    fn morozov_rejects_out_of_range_and_invalid() {
        let lambdas = [0.1, 0.2, 0.4];
        let residual = [0.5, 1.0, 1.5];
        // Target above the largest residual ⇒ unattainable.
        assert_eq!(morozov_lambda(&lambdas, &residual, 5.0, 1.0), None);
        // τ < 1 is invalid.
        assert_eq!(morozov_lambda(&lambdas, &residual, 1.0, 0.5), None);
    }
}
