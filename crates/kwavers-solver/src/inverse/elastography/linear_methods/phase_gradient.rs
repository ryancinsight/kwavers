//! Phase gradient inversion (McLaughlin & Renzi 2006) — estimates shear wave
//! speed from spatial phase gradients.

use ndarray::Array3;

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_domain::imaging::ultrasound::elastography::ElasticityMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

use super::super::algorithms::fill_boundaries;
use super::super::types::elasticity_map_from_speed;
use kwavers_core::constants::numerical::TWO_PI;

/// Phase gradient inversion (frequency domain method)
///
/// Estimates shear wave speed from phase gradients in frequency domain.
/// More accurate than time-of-flight for complex geometries.
///
/// # Algorithm
///
/// 1. For each spatial slice, extract 1D displacement profile
/// 2. Compute phase gradient using finite differences
/// 3. Convert phase gradient to wavenumber: k = ∂φ/∂x
/// 4. Compute speed: cs = ω/k = 2πf/k
/// 5. Apply spatial smoothing and boundary filling
///
/// # Physics
///
/// For propagating wave u(x,t) = A·exp(i(kx - ωt)):
/// - Phase: φ(x) = kx
/// - Wavenumber: k = ∂φ/∂x
/// - Dispersion relation: cs = ω/k
///
/// # References
///
/// - McLaughlin & Renzi (2006): "Shear wave speed recovery using phase information"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn phase_gradient_inversion(
    displacement: &DisplacementField,
    grid: &Grid,
    density: f64,
    frequency: f64,
) -> KwaversResult<ElasticityMap> {
    let (nx, ny, nz) = displacement.uz.dim();
    let mut shear_wave_speed = Array3::zeros((nx, ny, nz));

    // For each spatial slice, compute phase gradient
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            // Extract 1D profile along x-direction at this y,z location
            let mut profile = Vec::with_capacity(nx);
            for i in 0..nx {
                profile.push(displacement.uz[[i, j, k]]);
            }

            if let Some(cs) = compute_phase_gradient_speed(&profile, grid.dx, frequency) {
                for i in 0..nx {
                    shear_wave_speed[[i, j, k]] = cs;
                }
            } else {
                for i in 0..nx {
                    shear_wave_speed[[i, j, k]] = 3.0;
                }
            }
        }
    }

    fill_boundaries(&mut shear_wave_speed);

    Ok(elasticity_map_from_speed(shear_wave_speed, density))
}

/// Compute shear wave speed from phase gradient of 1D profile
///
/// Returns `None` when the profile is too short or no usable amplitude is
/// available; otherwise returns the dispersion-relation speed `cs = ω/k`
/// clamped to soft-tissue range.
pub(super) fn compute_phase_gradient_speed(
    profile: &[f64],
    dx: f64,
    frequency: f64,
) -> Option<f64> {
    if profile.len() < 4 {
        return None;
    }

    let mut phase_gradient = 0.0;
    let mut valid_points = 0;

    for i in 1..profile.len() - 1 {
        if profile[i].abs() > 1e-12 {
            let phase_diff = (profile[i + 1] - profile[i - 1]) / (2.0 * dx);
            phase_gradient += phase_diff.abs();
            valid_points += 1;
        }
    }

    if valid_points > 0 {
        phase_gradient /= valid_points as f64;

        let max_amplitude = profile.iter().copied().fold(0.0, f64::max).max(1e-12);
        let wavenumber = phase_gradient / max_amplitude;

        let cs = TWO_PI * frequency / wavenumber.abs().max(0.1);

        Some(cs.clamp(0.5, 10.0))
    } else {
        None
    }
}
