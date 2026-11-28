use super::{
    louisnard_gomez_growth_rate, louisnard_gomez_regime, LouisnardGomezParams,
    LOUISNARD_GOMEZ_MAX_FREQUENCY_HZ, LOUISNARD_GOMEZ_MAX_PRESSURE_PA,
    LOUISNARD_GOMEZ_MAX_RADIUS_M, LOUISNARD_GOMEZ_MIN_FREQUENCY_HZ,
    LOUISNARD_GOMEZ_MIN_PRESSURE_PA,
};
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};

/// Checked form of [`louisnard_gomez_growth_rate`]: same growth rate
/// [m/s], but rejects inputs outside the Louisnard & Gomez validated
/// regime instead of silently returning an extrapolation.
///
/// # Errors
/// - [`KwaversError::Physics`] with `PhysicsError::InvalidParameter`
///   when any input is degenerate (non-positive `R0`, `D`, `C0`, `T`,
///   `p0`; fewer than 2 history samples), or when
///   [`louisnard_gomez_regime`] reports the frequency, pressure,
///   radius, or saturation ratio outside the validated range
///   (20–100 kHz, 1–5 bar, `R0 < 15 µm`, `0 < C∞* ≤ 1`).
///   `acoustic_pressure` is the drive amplitude [Pa] the regime bounds
///   refer to; it is not otherwise used by the law, which is a
///   functional of the radius history.
pub fn louisnard_gomez_growth_rate_checked(
    radius: &[f64],
    time: &[f64],
    equilibrium_radius: f64,
    saturation_ratio: f64,
    acoustic_pressure: f64,
    frequency: f64,
    params: &LouisnardGomezParams,
) -> KwaversResult<f64> {
    let regime = louisnard_gomez_regime(
        equilibrium_radius,
        acoustic_pressure,
        frequency,
        saturation_ratio,
    );
    let reject = |parameter: &str, value: f64, reason: String| {
        KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: parameter.to_owned(),
            value,
            reason,
        })
    };
    // Each reject carries an explicit NaN disjunct: a bare
    // `x <= 0.0` test is false for NaN and would wave it through —
    // the NaN-tightness convention of the regime flags and the
    // Bader checked variant.
    if equilibrium_radius <= 0.0 || equilibrium_radius.is_nan() {
        return Err(reject(
            "equilibrium_radius",
            equilibrium_radius,
            "Equilibrium radius must be positive".to_owned(),
        ));
    }
    // Medium-parameter rejects, per this function's documented
    // contract: the unchecked law silently returns 0.0 for degenerate
    // D / C0 / T / p0, which a checked variant must not wave through
    // as Ok(0.0). NaN-tight for the same reason as above.
    if params.diffusivity <= 0.0 || params.diffusivity.is_nan() {
        return Err(reject(
            "diffusivity",
            params.diffusivity,
            "Diffusivity must be positive".to_owned(),
        ));
    }
    if params.saturation_concentration <= 0.0 || params.saturation_concentration.is_nan() {
        return Err(reject(
            "saturation_concentration",
            params.saturation_concentration,
            "Saturation concentration must be positive".to_owned(),
        ));
    }
    if params.temperature <= 0.0 || params.temperature.is_nan() {
        return Err(reject(
            "temperature",
            params.temperature,
            "Temperature must be positive".to_owned(),
        ));
    }
    if params.ambient_pressure <= 0.0 || params.ambient_pressure.is_nan() {
        return Err(reject(
            "ambient_pressure",
            params.ambient_pressure,
            "Ambient pressure must be positive".to_owned(),
        ));
    }
    if radius.len() < 2 || radius.len() != time.len() {
        return Err(reject(
            "radius history",
            radius.len() as f64,
            "A radius history of at least 2 samples over one cycle is required".to_owned(),
        ));
    }
    if regime.frequency_outside_validated {
        return Err(reject(
            "frequency",
            frequency,
            format!(
                "Louisnard-Gomez law validated at {LOUISNARD_GOMEZ_MIN_FREQUENCY_HZ}..={LOUISNARD_GOMEZ_MAX_FREQUENCY_HZ} Hz; got {frequency}"
            ),
        ));
    }
    if regime.pressure_outside_validated {
        return Err(reject(
            "acoustic_pressure",
            acoustic_pressure,
            format!(
                "Louisnard-Gomez law validated at {LOUISNARD_GOMEZ_MIN_PRESSURE_PA}..={LOUISNARD_GOMEZ_MAX_PRESSURE_PA} Pa; got {acoustic_pressure}"
            ),
        ));
    }
    if regime.radius_outside_validated {
        return Err(reject(
            "equilibrium_radius",
            equilibrium_radius,
            format!(
                "Louisnard-Gomez law validated for R0 < {LOUISNARD_GOMEZ_MAX_RADIUS_M} m; got {equilibrium_radius}"
            ),
        ));
    }
    if regime.saturation_outside_validated {
        return Err(reject(
            "saturation_ratio",
            saturation_ratio,
            "Louisnard-Gomez law validated for near-saturated liquid, 0 < C* <= 1".to_owned(),
        ));
    }
    Ok(louisnard_gomez_growth_rate(
        radius,
        time,
        equilibrium_radius,
        saturation_ratio,
        params,
    ))
}
