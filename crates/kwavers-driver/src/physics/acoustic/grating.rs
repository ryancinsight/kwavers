//! Element pitch vs grating-lobe steering bounds + the ULA array-factor kernel.
//!
//! This submodule carries the spatial-sampling half of phased-array acoustics: how the
//! element pitch `d` bounds the steering freedom before a grating lobe appears in real
//! space (the `d ≤ λ/2` rule for full ±90° steering) and the ULA beam pattern that the
//! driver's channel count + connector pitch actually produce.
//
//! Each kernel is pure-math (`f64`-in/`f64`-out, no state, no cross-slice dep) so it
//! feeds straight into the slice facade's named `pub use` re-export chain.

/// Maximum grating-lobe-free steering angle (degrees) for an element pitch and wavelength.
///
/// `sin θ_max = λ/d − 1`: pitch ≤ λ/2 ⇒ 90° (full steering); pitch ≥ λ ⇒ 0° (grating lobe at
/// broadside); in between, the partial steering limit.
#[must_use]
pub fn max_grating_free_steer_deg(pitch_m: f64, lambda_m: f64) -> f64 {
    if pitch_m <= 0.0 {
        return 90.0;
    }
    let s = lambda_m / pitch_m - 1.0;
    if s >= 1.0 {
        90.0
    } else if s <= 0.0 {
        0.0
    } else {
        s.asin().to_degrees()
    }
}

/// Angle (degrees) of the first grating lobe when steering to `steer_deg`, if one exists in real
/// space (`|sin θ_g| ≤ 1`). `sin θ_g = sin θ_steer − λ/d`.
#[must_use]
pub fn grating_lobe_angle_deg(pitch_m: f64, lambda_m: f64, steer_deg: f64) -> Option<f64> {
    if pitch_m <= 0.0 {
        return None;
    }
    let sg = steer_deg.to_radians().sin() - lambda_m / pitch_m;
    if sg.abs() <= 1.0 {
        Some(sg.asin().to_degrees())
    } else {
        None
    }
}

/// Normalised array factor (0…1) of an `n`-element uniform linear array at pitch `d`, steered to
/// `steer_deg`, evaluated at `theta_deg`: `|sin(Nψ/2) / (N·sin(ψ/2))|`, `ψ = k·d·(sinθ − sinθ_s)`.
/// This is the beam the driver's channel count + connector pitch actually produce — peak at the
/// steer angle, nulls, side lobes, and full-height grating lobes when the pitch is too large.
#[must_use]
pub fn array_factor(n: usize, pitch_m: f64, lambda_m: f64, steer_deg: f64, theta_deg: f64) -> f64 {
    if n == 0 || lambda_m <= 0.0 {
        return 0.0;
    }
    let k = 2.0 * std::f64::consts::PI / lambda_m;
    let psi = k * pitch_m * (theta_deg.to_radians().sin() - steer_deg.to_radians().sin());
    let half = psi / 2.0;
    let s = half.sin();
    if s.abs() < 1.0e-12 {
        1.0 // main lobe (or a grating lobe): ψ a multiple of 2π
    } else {
        ((n as f64 * half).sin() / (n as f64 * s)).abs()
    }
}
