//! Nonlinear shock-parameter (Earnshaw): the harmonic-distortion indicator for the focal regime.
//!
//! This submodule carries the single nonlinear-propagation kernel: the normalised shock
//! parameter `σ = z / z_shock` that flags whether the focal field has accumulated enough
//! nonlinear steepening to be in the shocked-harmonic regime.
//!
//! Pure-math (`f64`-in/`f64`-out, no state, no cross-slice dep). The slice facade re-exports
//! it under the canonical name `nonlinear_shock_parameter`.

/// Normalised nonlinear shock parameter σ at propagation distance `z` for a sinusoidal source.
///
/// `σ = z / z_shock`,  `z_shock = ρ·c³ / (β·ω·p₀)` (plane-wave Earnshaw shock distance),
/// where `β = 1 + B/(2A)` is the nonlinearity parameter (water ≈ 3.5, tissue ≈ 4.0),
/// `ω = 2π·f`, `p₀` the source peak pressure.
///
/// When `σ < 1` the wave is quasi-linear; at `σ ≈ 1` a shock forms; for `σ > 1` the waveform
/// is fully shocked with strong harmonic content. High-intensity neuromodulation sits in the
/// `σ ≈ 0.5–2` regime — relevant for harmonic imaging and nonlinear focal gain.
///
/// Returns `+∞` when shock distance is zero (zero frequency or zero source pressure).
#[must_use]
pub fn nonlinear_shock_parameter(
    p0_pa: f64,
    freq_hz: f64,
    z_m: f64,
    rho_kg_m3: f64,
    speed_m_s: f64,
    b_over_a: f64,
) -> f64 {
    if p0_pa <= 0.0 || freq_hz <= 0.0 || speed_m_s <= 0.0 || rho_kg_m3 <= 0.0 {
        return f64::INFINITY;
    }
    let beta = 1.0 + b_over_a / 2.0;
    let omega = 2.0 * std::f64::consts::PI * freq_hz;
    let z_shock = rho_kg_m3 * speed_m_s.powi(3) / (beta * omega * p0_pa);
    z_m / z_shock
}
