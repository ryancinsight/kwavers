//! Shared analytical helpers for nonlinear elastography inversion.
//!
//! All formulas are from Rénier et al. (2008) JASA 124(5) and
//! Destrade & Ogden (2010) JASA 128(6).

use super::super::config::NonlinearInversionConfig;
use crate::core::constants::numerical::{TWO_PI};

/// μ = ρ c_s²  (Hooke's law for shear waves)
#[must_use]
#[inline]
pub(super) fn shear_modulus(config: &NonlinearInversionConfig) -> f64 {
    let c_s = config.shear_wave_speed.max(1e-3);
    config.density * c_s * c_s
}

/// β_s from measured displacement amplitudes (Rénier 2008, Eq. 8):
///
/// ```text
/// β_s = 2 A₂ c_s / (ω A₁² z)
/// ```
///
/// Returns `None` if A₁ < 1e-12 m (below noise floor).
#[inline]
pub(super) fn beta_s_from_amplitudes(
    a1: f64,
    a2: f64,
    config: &NonlinearInversionConfig,
) -> Option<f64> {
    if a1 < 1e-12 {
        return None;
    }
    let omega = TWO_PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed.max(1e-3);
    let z = config.propagation_distance.max(1e-6);
    Some(2.0 * a2 * c_s / (omega * a1 * a1 * z))
}

/// B/A = 2(β_s − 1)  [acoustic convention; linear medium → 0]
#[must_use]
#[inline]
pub(super) fn ba_from_beta_s(beta_s: f64) -> f64 {
    2.0 * (beta_s - 1.0)
}

/// A_L = μ(4 β_s − 3)  (Destrade & Ogden 2010, Eq. 3.8)
#[must_use]
#[inline]
pub(super) fn a_landau(mu: f64, beta_s: f64) -> f64 {
    mu * 4.0f64.mul_add(beta_s, -3.0)
}

/// Forward model: predict (A₁_pred, A₂_pred) given B/A and observed A₁.
///
/// ```text
/// A₁_pred = a1_obs
/// A₂_pred = β_s k_s a1² z / 2   [Rénier 2008, Eq. 7]
/// ```
#[must_use]
pub(super) fn forward_model(
    ba_ratio: f64,
    a1_obs: f64,
    config: &NonlinearInversionConfig,
) -> (f64, f64) {
    let omega = TWO_PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed.max(1e-3);
    let k_s = omega / c_s;
    let z = config.propagation_distance.max(1e-6);
    let beta = ba_ratio / 2.0 + 1.0;
    let a2_pred = (beta * k_s * a1_obs * a1_obs * z / 2.0).max(0.0);
    (a1_obs, a2_pred)
}

/// Jacobian of forward model w.r.t. B/A:
///
/// ```text
/// ∂A₁_pred / ∂(B/A) = 0
/// ∂A₂_pred / ∂(B/A) = k_s A₁² z / 4   (∂β_s/∂(B/A) = 1/2)
/// ```
#[must_use]
pub(super) fn forward_model_derivative(
    _ba_ratio: f64,
    a1_obs: f64,
    config: &NonlinearInversionConfig,
) -> (f64, f64) {
    let omega = TWO_PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed.max(1e-3);
    let k_s = omega / c_s;
    let z = config.propagation_distance.max(1e-6);
    let da2_dba = k_s * a1_obs * a1_obs * z / 4.0;
    (0.0_f64, da2_dba)
}
