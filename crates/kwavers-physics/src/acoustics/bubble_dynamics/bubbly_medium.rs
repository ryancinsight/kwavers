//! Acoustic properties of a bubbly liquid as a function of gas void fraction.
//!
//! A residual microbubble cloud changes the effective medium the next pulse
//! propagates through: even a tiny gas void fraction collapses the sound speed
//! (Wood 1930) — creating an impedance mismatch that reflects the beam — and
//! adds resonant-scattering attenuation (Commander & Prosperetti 1989). These
//! are the genuine physical mechanisms behind residual-bubble shielding.

use kwavers_core::constants::numerical::TWO_PI;
use std::f64::consts::PI;

/// Volume-weighted mixture density of a bubbly liquid.
///
/// `ρ_m = (1 − β)·ρ_liquid + β·ρ_gas` for gas void fraction `β ∈ [0, 1)`.
#[must_use]
#[inline]
pub fn mixture_density(void_fraction: f64, rho_liquid: f64, rho_gas: f64) -> f64 {
    let b = void_fraction.clamp(0.0, 1.0);
    (1.0 - b).mul_add(rho_liquid, b * rho_gas)
}

/// Low-frequency sound speed of a bubbly liquid (Wood 1930).
///
/// ## Theorem
/// The mixture is mechanically a series of liquid and gas compressibilities and
/// a parallel mass:
/// ```text
///   κ_m = (1 − β)·κ_liquid + β·κ_gas,   κ = 1/(ρ c²)
///   ρ_m = (1 − β)·ρ_liquid + β·ρ_gas
///   c_m = 1 / √(ρ_m · κ_m)
/// ```
/// Because `κ_gas ≫ κ_liquid` (gas is ~10⁴× more compressible), even a void
/// fraction of `10⁻⁴` drops `c_m` from ~1500 m/s toward a few hundred m/s — the
/// dramatic impedance change that reflects/refracts a subsequent pulse.
///
/// # Arguments
/// * `void_fraction` – gas volume fraction `β` [-]
/// * `c_liquid`, `rho_liquid` – liquid sound speed [m/s] and density [kg/m³]
/// * `c_gas`, `rho_gas` – gas sound speed [m/s] and density [kg/m³]
///
/// Returns the mixture sound speed [m/s]; falls back to `c_liquid` for
/// non-physical inputs.
#[must_use]
pub fn wood_sound_speed(
    void_fraction: f64,
    c_liquid: f64,
    rho_liquid: f64,
    c_gas: f64,
    rho_gas: f64,
) -> f64 {
    if !(c_liquid > 0.0 && rho_liquid > 0.0 && c_gas > 0.0 && rho_gas > 0.0) {
        return c_liquid.max(0.0);
    }
    let b = void_fraction.clamp(0.0, 1.0 - 1e-12);
    let kappa_l = 1.0 / (rho_liquid * c_liquid * c_liquid);
    let kappa_g = 1.0 / (rho_gas * c_gas * c_gas);
    let kappa_m = (1.0 - b).mul_add(kappa_l, b * kappa_g);
    let rho_m = mixture_density(b, rho_liquid, rho_gas);
    1.0 / (rho_m * kappa_m).sqrt()
}

/// Complex mixture wavenumber `k_m = Re(k_m) + i·Im(k_m)` [1/m] of a
/// monodisperse bubble cloud (Commander & Prosperetti 1989).
///
/// ## Model
/// For a monodisperse cloud of number density `N = β / ((4/3)π R₀³)`,
/// ```text
///   k_m² = (ω/c_l)² + 4π N R₀ ω² / (ω₀² − ω² + 2 i b ω)
///   k_m  = √(k_m²)   (principal branch)
/// ```
/// with Minnaert resonance `ω₀² = 3κP₀/(ρ_l R₀²)` and total damping
/// `b = ½ ω₀ δ`, `δ = δ_radiation + δ_viscous` (thermal damping omitted as a
/// conservative lower bound on `δ`). The **imaginary** part is the amplitude
/// attenuation `α(ω)` [Np/m]; the **real** part sets the frequency-dependent
/// phase velocity `c_p(ω) = ω / Re(k_m)` — together these are the
/// Kramers–Kronig-consistent dispersive response of the cloud.
///
/// Returns `(Re k_m, Im k_m)`; the non-dispersive `(ω/c_l, 0)` for non-physical
/// inputs or zero void fraction.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn commander_prosperetti_wavenumber(
    freq_hz: f64,
    void_fraction: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> (f64, f64) {
    let b = void_fraction.clamp(0.0, 1.0 - 1e-12);
    let kl = if c_liquid > 0.0 && freq_hz > 0.0 {
        TWO_PI * freq_hz / c_liquid
    } else {
        0.0
    };
    if !(freq_hz > 0.0 && r0_m > 0.0 && c_liquid > 0.0 && rho_liquid > 0.0 && p0_pa > 0.0)
        || b <= 0.0
    {
        return (kl, 0.0);
    }
    let omega = TWO_PI * freq_hz;
    // Minnaert resonance angular frequency.
    let omega0_sq = 3.0 * polytropic * p0_pa / (rho_liquid * r0_m * r0_m);
    let omega0 = omega0_sq.sqrt();
    // Dimensionless damping: radiation + viscous (thermal omitted, lower bound).
    let delta_rad = omega0 * r0_m / c_liquid;
    let delta_visc = 4.0 * mu_liquid / (rho_liquid * omega0 * r0_m * r0_m);
    let damping_b = 0.5 * omega0 * (delta_rad + delta_visc);

    let number_density = b / ((4.0 / 3.0) * PI * r0_m.powi(3));
    // k_m² = k_l² + 4π N R₀ ω² / (ω₀² − ω² + 2 i b ω).
    let kl_sq = kl * kl;
    let denom_re = omega0_sq - omega * omega;
    let denom_im = 2.0 * damping_b * omega;
    let denom_mag_sq = denom_re * denom_re + denom_im * denom_im;
    let prefac = 4.0 * PI * number_density * r0_m * omega * omega;
    // Complex k_m² = (kl_sq + prefac·denom_re/|denom|²) + i·(−prefac·denom_im/|denom|²).
    let km2_re = kl_sq + prefac * denom_re / denom_mag_sq;
    let km2_im = -prefac * denom_im / denom_mag_sq;
    // √ of a complex number a+ib: re = √((|z|+a)/2), im = sign(b)·√((|z|−a)/2).
    let mag = (km2_re * km2_re + km2_im * km2_im).sqrt();
    let km_re = (0.5 * (mag + km2_re)).max(0.0).sqrt();
    let km_im = (0.5 * (mag - km2_re)).max(0.0).sqrt();
    // Principal branch with km2_im ≤ 0 gives a decaying wave (Im k_m ≥ 0 as an
    // attenuation magnitude); km_re > 0 is the propagating part.
    (km_re, km_im)
}

/// Resonant-scattering attenuation of a monodisperse bubble cloud
/// (Commander & Prosperetti 1989): the amplitude attenuation coefficient
/// `α = Im(k_m)` [Np/m] at frequency `f`. See
/// [`commander_prosperetti_wavenumber`] for the full model.
///
/// Returns `α` [Np/m] (≥ 0); `0` for non-physical inputs.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn commander_prosperetti_attenuation(
    freq_hz: f64,
    void_fraction: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> f64 {
    commander_prosperetti_wavenumber(
        freq_hz,
        void_fraction,
        r0_m,
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
    )
    .1
}

/// Frequency-dependent phase velocity `c_p(ω) = ω / Re(k_m)` [m/s] of a
/// monodisperse bubble cloud (the dispersive companion of the attenuation;
/// Commander & Prosperetti 1989). Near resonance the phase velocity departs
/// strongly from both the host `c_l` and the (zero-frequency) Wood limit —
/// anomalous dispersion — which is why a complete broadband model carries this
/// term in addition to attenuation.
///
/// Returns `c_p` [m/s]; falls back to `c_liquid` for non-physical inputs or zero
/// void fraction.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn commander_prosperetti_phase_velocity(
    freq_hz: f64,
    void_fraction: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> f64 {
    let (km_re, _) = commander_prosperetti_wavenumber(
        freq_hz,
        void_fraction,
        r0_m,
        c_liquid,
        rho_liquid,
        mu_liquid,
        p0_pa,
        polytropic,
    );
    if km_re > 0.0 && freq_hz > 0.0 {
        TWO_PI * freq_hz / km_re
    } else {
        c_liquid.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const C_W: f64 = 1481.0;
    const RHO_W: f64 = 998.0;
    const C_AIR: f64 = 343.0;
    const RHO_AIR: f64 = 1.2;
    const MU_W: f64 = 1.0e-3;
    const P0: f64 = 101_325.0;

    #[test]
    fn wood_zero_void_is_liquid_sound_speed() {
        let c = wood_sound_speed(0.0, C_W, RHO_W, C_AIR, RHO_AIR);
        assert!((c - C_W).abs() < 1e-6, "β=0 must give c_liquid; got {c}");
    }

    #[test]
    fn wood_tiny_void_collapses_sound_speed() {
        // Classic Wood result: β = 1e-4 already roughly halves c (~930 m/s), and
        // β = 1e-3 drops it below 500 m/s.
        let c4 = wood_sound_speed(1e-4, C_W, RHO_W, C_AIR, RHO_AIR);
        assert!(
            c4 < 1000.0 && c4 > 0.0,
            "β=1e-4 should roughly halve c; got {c4}"
        );
        let c3 = wood_sound_speed(1e-3, C_W, RHO_W, C_AIR, RHO_AIR);
        assert!(
            c3 < 500.0,
            "β=1e-3 should collapse c below 500 m/s; got {c3}"
        );
        // Minimum of the Wood curve is well below either pure phase.
        let c_mid = wood_sound_speed(0.5, C_W, RHO_W, C_AIR, RHO_AIR);
        assert!(c_mid < C_AIR, "mixture c should dip below pure-gas c");
    }

    #[test]
    fn wood_monotone_drop_then_rise() {
        // c decreases from β=0 to the minimum (small β) — verify the initial drop.
        let c0 = wood_sound_speed(0.0, C_W, RHO_W, C_AIR, RHO_AIR);
        let c1 = wood_sound_speed(1e-5, C_W, RHO_W, C_AIR, RHO_AIR);
        let c2 = wood_sound_speed(1e-3, C_W, RHO_W, C_AIR, RHO_AIR);
        assert!(c1 < c0 && c2 < c1, "c must drop with rising β at small β");
    }

    #[test]
    fn attenuation_zero_at_zero_void() {
        let a = commander_prosperetti_attenuation(1e6, 0.0, 2e-6, C_W, RHO_W, MU_W, P0, 1.4);
        assert_eq!(a, 0.0, "no bubbles ⇒ no excess attenuation");
    }

    #[test]
    fn attenuation_increases_with_void_fraction() {
        let a_lo = commander_prosperetti_attenuation(1e6, 1e-6, 2e-6, C_W, RHO_W, MU_W, P0, 1.4);
        let a_hi = commander_prosperetti_attenuation(1e6, 1e-4, 2e-6, C_W, RHO_W, MU_W, P0, 1.4);
        assert!(
            a_hi > a_lo && a_lo > 0.0,
            "α must grow with β: lo={a_lo}, hi={a_hi}"
        );
        assert!(a_hi.is_finite());
    }

    #[test]
    fn phase_velocity_equals_liquid_without_bubbles() {
        let cp = commander_prosperetti_phase_velocity(1e6, 0.0, 2e-6, C_W, RHO_W, MU_W, P0, 1.4);
        assert!((cp - C_W).abs() < 1e-6, "β=0 ⇒ c_p = c_liquid; got {cp}");
    }

    #[test]
    fn phase_velocity_disperses_across_resonance() {
        // Below resonance the cloud slows the wave (c_p < c_l, toward Wood); above
        // resonance the phase velocity exceeds c_l — anomalous dispersion. The two
        // regimes must straddle c_l, which a single non-dispersive speed cannot do.
        let r0 = 3e-6;
        let beta = 1e-4;
        let omega0 = (3.0 * 1.4 * P0 / (RHO_W * r0 * r0)).sqrt();
        let f_res = omega0 / (2.0 * std::f64::consts::PI);
        let cp_below =
            commander_prosperetti_phase_velocity(f_res * 0.5, beta, r0, C_W, RHO_W, MU_W, P0, 1.4);
        let cp_above =
            commander_prosperetti_phase_velocity(f_res * 2.0, beta, r0, C_W, RHO_W, MU_W, P0, 1.4);
        assert!(
            cp_below < C_W,
            "below resonance the cloud must slow the wave: c_p={cp_below}, c_l={C_W}"
        );
        assert!(
            cp_above > C_W,
            "above resonance the phase velocity must exceed c_l (anomalous): c_p={cp_above}"
        );
    }

    #[test]
    fn phase_velocity_low_frequency_limit_matches_wood() {
        // The ω→0 limit of the CP phase velocity must reproduce the Wood mixture
        // sound speed (to first order in β). This is why the dispersion operator
        // subsumes the Wood collapse rather than double-counting it.
        let r0 = 2e-6;
        let beta = 1e-5; // dilute, so the first-order match is tight
                         // Resonance ≈ 1.6 MHz; evaluate well below it.
        let f_low = 1.0e3;
        let cp = commander_prosperetti_phase_velocity(f_low, beta, r0, C_W, RHO_W, MU_W, P0, 1.4);
        let c_wood = wood_sound_speed(beta, C_W, RHO_W, C_AIR, RHO_AIR);
        let rel = (cp - c_wood).abs() / c_wood;
        assert!(
            rel < 0.05,
            "low-frequency CP phase velocity must approach Wood: c_p={cp:.2}, c_wood={c_wood:.2} (rel {rel:.3})"
        );
    }

    #[test]
    fn wavenumber_imag_matches_attenuation() {
        // SSOT: the attenuation wrapper returns exactly Im(k_m).
        let (_, km_im) =
            commander_prosperetti_wavenumber(1e6, 1e-4, 2e-6, C_W, RHO_W, MU_W, P0, 1.4);
        let alpha = commander_prosperetti_attenuation(1e6, 1e-4, 2e-6, C_W, RHO_W, MU_W, P0, 1.4);
        assert!((km_im - alpha).abs() < 1e-15, "Im(k_m) must equal α");
        assert!(alpha > 0.0);
    }

    #[test]
    fn attenuation_peaks_near_resonance() {
        // Resonance of a 3 µm air bubble ≈ 1.1 MHz; attenuation near resonance
        // exceeds that far below it.
        let r0 = 3e-6;
        let near = commander_prosperetti_attenuation(1.1e6, 1e-5, r0, C_W, RHO_W, MU_W, P0, 1.4);
        let below = commander_prosperetti_attenuation(0.1e6, 1e-5, r0, C_W, RHO_W, MU_W, P0, 1.4);
        assert!(
            near > below,
            "α should peak near resonance: near={near}, below={below}"
        );
    }
}
