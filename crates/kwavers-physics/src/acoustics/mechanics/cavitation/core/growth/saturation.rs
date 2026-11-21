//! Independent far-field and saturation concentrations for steady diffusion.

use super::{cycle_transport, FOUR_PI, RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO};
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};

/// Signed steady molar flux into a spherical bubble in water [mol/s].
///
/// Both concentrations refer to the same gas and liquid temperature.
/// `saturation_concentration` is the dissolved concentration in
/// equilibrium with a flat interface at the ambient pressure.
/// `far_field_concentration` is the actual dissolved concentration in the bulk.
/// Following Zhang (2012), Eq. (3.6), the independently specified bulk
/// concentration adds `4πR D ⟨R(t)/R⟩ (C_i − C_0)` to the saturated flux.
///
/// The model assumes sinusoidal forcing and small fractional radius oscillation.
/// Check the amplitude against the accuracy required by the application.
///
/// # Errors
///
/// Returns a physics error for non-finite inputs, negative acoustic amplitude,
/// frequency, diffusivity, or bulk concentration, or non-positive radius,
/// ambient pressure, or saturation concentration. Rejects a pressure ratio
/// above the legacy guard, fractional amplitudes at least unity, and
/// non-finite computed flux.
pub fn rectified_diffusion_rate_with_concentrations(
    radius: f64,
    ambient_pressure: f64,
    acoustic_pressure: f64,
    frequency: f64,
    diffusivity: f64,
    saturation_concentration: f64,
    far_field_concentration: f64,
) -> KwaversResult<f64> {
    let parameters = [
        ("radius", radius, radius > 0.0),
        ("ambient_pressure", ambient_pressure, ambient_pressure > 0.0),
        (
            "acoustic_pressure",
            acoustic_pressure,
            acoustic_pressure >= 0.0,
        ),
        ("frequency", frequency, frequency >= 0.0),
        ("diffusivity", diffusivity, diffusivity >= 0.0),
        (
            "saturation_concentration",
            saturation_concentration,
            saturation_concentration > 0.0,
        ),
        (
            "far_field_concentration",
            far_field_concentration,
            far_field_concentration >= 0.0,
        ),
    ];
    if let Some((parameter, value, _)) = parameters
        .into_iter()
        .find(|(_, value, in_range)| !value.is_finite() || !in_range)
    {
        return Err(invalid(
            parameter,
            value,
            "Expected a finite value in the specified physical range",
        ));
    }
    let pressure_ratio = acoustic_pressure / ambient_pressure;
    if pressure_ratio > RECTIFIED_DIFFUSION_MAX_PRESSURE_RATIO {
        return Err(invalid(
            "acoustic_pressure / ambient_pressure",
            pressure_ratio,
            "The pressure ratio exceeds the steady diffusion guard",
        ));
    }
    let cycle = cycle_transport(radius, ambient_pressure, acoustic_pressure, frequency);
    if !cycle.epsilon.is_finite() || cycle.epsilon >= 1.0 {
        return Err(invalid(
            "epsilon",
            cycle.epsilon,
            "Expected fractional oscillation amplitude below unity",
        ));
    }
    // Compute the signed concentration jump directly, including near equilibrium.
    let jump = (-saturation_concentration).mul_add(cycle.pressure_ratio, far_field_concentration);
    let net_flux = FOUR_PI * radius * diffusivity * cycle.mean_radius * jump;
    if !net_flux.is_finite() {
        return Err(invalid(
            "net_flux",
            net_flux,
            "Computed flux must be finite",
        ));
    }
    Ok(net_flux)
}

fn invalid(parameter: &str, value: f64, reason: &str) -> KwaversError {
    KwaversError::Physics(PhysicsError::InvalidParameter {
        parameter: parameter.to_owned(),
        value,
        reason: reason.to_owned(),
    })
}

#[cfg(test)]
mod tests;
