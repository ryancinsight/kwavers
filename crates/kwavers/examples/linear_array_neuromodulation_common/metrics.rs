//! Scalar metrics for focal fields: widths, gains, mechanical index, shock distance.

use super::{B_OVER_A, DENSITY_KG_M3, SOUND_SPEED_M_S};
use eunomia::FloatElement;

/// Index of the maximum of a profile.
#[must_use]
pub fn peak_index(profile: &[f64]) -> usize {
    profile
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0
}

/// Full width at half maximum of a profile about its peak, in the profile's units.
///
/// Crossings are linearly interpolated on each side of the peak. Returns `None`
/// if the profile does not fall below half of its peak on both sides.
#[must_use]
pub fn fwhm(profile: &[f64], spacing: f64) -> Option<f64> {
    let peak = peak_index(profile);
    let half = 0.5 * profile[peak];
    let left = (0..peak).rev().find(|&index| profile[index] < half)?;
    let right = (peak + 1..profile.len()).find(|&index| profile[index] < half)?;
    let cross = |a: usize, b: usize| {
        let fraction = (half - profile[a]) / (profile[b] - profile[a]);
        f64::from_count(a) + fraction * (f64::from_count(b) - f64::from_count(a))
    };
    let left_position = cross(left + 1, left);
    let right_position = cross(right - 1, right);
    Some((right_position - left_position) * spacing)
}

/// Mechanical index `MI = p_neg[MPa] / sqrt(f[MHz])`.
#[must_use]
pub fn mechanical_index(peak_negative_pa: f64, frequency_hz: f64) -> f64 {
    (peak_negative_pa.abs() * 1.0e-6) / (frequency_hz * 1.0e-6).sqrt()
}

/// Surface pressure needed to reach `target_pa` at a focus with pressure gain `gain`.
#[must_use]
pub fn required_surface_pa(target_pa: f64, gain: f64) -> f64 {
    target_pa / gain
}

/// Nonlinear shock-formation distance for a plane wave of peak pressure `p0` at `frequency_hz`.
///
/// `x_s = ρ c³ / (β ω p0)` with `β = 1 + B/(2A)` (Hamilton & Blackstock). Used as a
/// bound on where nonlinear distortion becomes significant along the beam.
#[must_use]
pub fn shock_distance_m(p0_pa: f64, frequency_hz: f64) -> f64 {
    let beta = 1.0 + 0.5 * B_OVER_A;
    let omega = std::f64::consts::TAU * frequency_hz;
    DENSITY_KG_M3 * SOUND_SPEED_M_S.powi(3) / (beta * omega * p0_pa)
}
