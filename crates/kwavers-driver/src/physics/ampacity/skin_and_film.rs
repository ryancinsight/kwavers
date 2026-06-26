/// Copper skin depth (m) at frequency `f` (Hz): `δ = √(ρ / (π·f·μ₀))` (copper µ_r ≈ 1).
#[must_use]
pub fn skin_depth_m(freq_hz: f64) -> f64 {
    if freq_hz <= 0.0 {
        return f64::INFINITY;
    }
    let rho = 1.68e-8;
    let mu0 = 1.256_637_062e-6;
    (rho / (std::f64::consts::PI * freq_hz * mu0)).sqrt()
}

/// AC/DC resistance ratio for a foil of thickness `t` carrying current on both faces, at frequency
/// `f`. For `t ≪ δ` the ratio → 1 (field fully penetrates); for `t ≫ δ` it grows ~`t/(2δ)`.
#[must_use]
pub fn ac_resistance_factor(thickness_m: f64, freq_hz: f64) -> f64 {
    let delta = skin_depth_m(freq_hz);
    if !delta.is_finite() || delta <= 0.0 {
        return 1.0;
    }
    // Effective conduction depth from both faces: t·tanh(t/2δ)/(t/2δ)·… — use the standard foil
    // model R_ac/R_dc = (t/2δ) · coth(t/2δ), clamped to ≥ 1.
    let u = thickness_m / (2.0 * delta);
    if u < 1.0e-3 {
        1.0
    } else {
        (u / u.tanh()).max(1.0)
    }
}
