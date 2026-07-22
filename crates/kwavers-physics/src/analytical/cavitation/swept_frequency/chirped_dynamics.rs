//! Chirped-drive Keller–Miksis bubble dynamics.
//!
//! Integrates a bubble under a frequency-swept ([`FrequencySweep`]) drive using
//! the same audited Keller–Miksis wall-pressure balance and RK4 loop as the
//! monochromatic solver — the only difference is the acoustic forcing closure,
//! which here is the chirp `(p_ac, ṗ_ac)` instead of a single sinusoid. The
//! shared core [`keller_miksis_forced_rk4`] guarantees one source of truth for
//! the dynamics.

use super::super::dynamics::{keller_miksis_forced_rk4, KmShellParams};
use super::chirp::FrequencySweep;

/// Integrate the shell-damped Keller–Miksis equation under a chirped drive.
///
/// Identical physics to [`super::super::keller_miksis_shelled_rk4`] but the
/// far-field acoustic pressure is the swept waveform
/// `p_ac(t) = A·sin φ(t)`, with exact instantaneous frequency `f(t)` and phase
/// `φ(t)` supplied by `sweep`. `xi_s = 0` gives a bare bubble.
///
/// # Arguments
/// * `sweep` – frequency-swept drive (carrier).
/// * `amplitude_pa` – peak acoustic pressure `A` `Pa`.
/// * `r0_m`, `rdot0` – initial radius `m` and wall velocity [m/s].
/// * `t_arr` – strictly increasing time samples `s`.
/// * `p0_pa`, `rho`, `sigma`, `mu`, `kappa`, `p_v_pa`, `xi_s`, `c_liquid` –
///   ambient pressure, liquid density, surface tension, liquid viscosity,
///   polytropic exponent, vapor pressure, shell viscosity, sound speed.
///
/// # Returns
/// `(R(t), Ṙ(t))` over `t_arr`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn chirped_keller_miksis_rk4(
    sweep: &FrequencySweep,
    amplitude_pa: f64,
    r0_m: f64,
    rdot0: f64,
    t_arr: &[f64],
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    xi_s: f64,
    c_liquid: f64,
) -> (Vec<f64>, Vec<f64>) {
    let params = KmShellParams {
        r0_m,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        p_v_pa,
        xi_s,
        c_liquid,
    };
    keller_miksis_forced_rk4(params, rdot0, t_arr, move |t_ret| {
        sweep.forcing(t_ret, amplitude_pa)
    })
}

/// Peak expansion ratio `R_max / R₀` of a bubble under a chirped drive — the
/// inertial-cavitation discriminant. `R_max/R₀ ≳ 2` is the standard violent
/// (inertial) collapse criterion (Flynn 1964; Apfel & Holland 1991): the bubble
/// expands well past equilibrium and collapses inertially.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn chirped_peak_expansion_ratio(
    sweep: &FrequencySweep,
    amplitude_pa: f64,
    r0_m: f64,
    t_arr: &[f64],
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    xi_s: f64,
    c_liquid: f64,
) -> f64 {
    if r0_m <= 0.0 {
        return 0.0;
    }
    let (r, _) = chirped_keller_miksis_rk4(
        sweep,
        amplitude_pa,
        r0_m,
        0.0,
        t_arr,
        p0_pa,
        rho,
        sigma,
        mu,
        kappa,
        p_v_pa,
        xi_s,
        c_liquid,
    );
    let r_max = r.iter().copied().fold(r0_m, f64::max);
    r_max / r0_m
}
