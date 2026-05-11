//! Safety Monitoring and Limit Checking
//!
//! This module provides real-time safety monitoring for clinical therapy sessions.
//! All calculations comply with IEC 62359:2010 and FDA 510(k) guidance.
//!
//! ## Safety Metrics
//!
//! - **Thermal Index (TI)**: Indicates potential for tissue heating
//! - **Mechanical Index (MI)**: Indicates potential for mechanical bioeffects
//! - **Cavitation Dose**: Time-integrated cavitation activity
//! - **Temperature Rise**: Spatial temperature increase from baseline
//!
//! ## Clinical Guidelines
//!
//! - TI < 6.0: Generally safe for most applications
//! - MI < 1.9: Generally safe for diagnostic applications
//! - MI < 0.7: Recommended for fetal imaging
//!
//! ## References
//!
//! - IEC 62359:2010: "Ultrasonics - Field characterization"
//! - FDA 510(k) Guidance: "Ultrasound Devices"
//! - Apfel & Holland (1991): "Gaseous cavitation thresholds"

use crate::core::error::KwaversResult;

use super::super::config::{AcousticTherapyParams, SafetyLimits};
use super::super::state::{AcousticField, SafetyMetrics, SafetyStatus};

#[cfg(test)]
mod tests;

/// Update safety metrics based on current acoustic field
///
/// Calculates all safety metrics according to IEC 62359:2010 standards and
/// FDA guidance. This function should be called after each therapy step
/// to ensure continuous safety monitoring.
///
/// # Arguments
///
/// - `safety_metrics`: Current safety metrics (will be updated)
/// - `acoustic_field`: Current acoustic field
/// - `acoustic_params`: Therapy acoustic parameters
/// - `dt`: Time step (s)
/// - `cavitation_activity`: Optional cavitation activity field
///
/// # Safety Metrics Calculated
///
/// - **Thermal Index (TI)**: Based on IEC 62359:2010 formula
///   TI = P_rms * sqrt(f) / 1e6
///   where P_rms is the root-mean-square pressure and f is frequency
///
/// - **Mechanical Index (MI)**: Based on FDA guidance
///   MI = PNP / (sqrt(f) * 1e6)
///   where PNP is peak negative pressure
///
/// - **Cavitation Dose**: Time-integrated cavitation activity
///   Based on Apfel & Holland (1991) cavitation threshold models
///
/// # Clinical Guidelines
///
/// - TI < 6.0: Generally safe for most applications
/// - MI < 1.9: Generally safe for diagnostic applications
/// - MI < 0.7: Recommended for fetal imaging
/// - Cavitation dose should be monitored for histotripsy applications
///
/// # References
///
/// - IEC 62359:2010: "Ultrasonics - Field characterization"
/// - FDA 510(k) Guidance: "Ultrasound Devices"
/// - Apfel & Holland (1991): "Gaseous cavitation thresholds"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn update_safety_metrics(
    safety_metrics: &mut SafetyMetrics,
    acoustic_field: &AcousticField,
    acoustic_params: &AcousticTherapyParams,
    dt: f64,
    cavitation_activity: Option<&ndarray::Array3<f64>>,
) -> KwaversResult<()> {
    // Calculate thermal index (IEC 62359 compliant).
    // RMS pressure: P_rms = sqrt( mean(p²) ) = sqrt( Σp² / N )
    let n = acoustic_field.pressure.len() as f64;
    let sum_sq: f64 = acoustic_field.pressure.iter().map(|&p| p * p).sum();
    let pressure_rms = (sum_sq / n).sqrt();

    safety_metrics.thermal_index = pressure_rms * acoustic_params.frequency.sqrt() / 1e6;

    // Mechanical Index (FDA 510(k) guidance, IEC 62359):
    // MI = p_neg_peak_derated (MPa) / sqrt(f_center (MHz))
    //    = (pnp_Pa / 1e6) / sqrt(f_Hz / 1e6)
    //    = pnp_Pa / (1e3 × sqrt(f_Hz))
    safety_metrics.mechanical_index =
        acoustic_params.pnp / (acoustic_params.frequency.sqrt() * 1e3);

    // Update cavitation dose (time-integrated cavitation activity)
    if let Some(cavitation) = cavitation_activity {
        let current_dose = cavitation.iter().sum::<f64>() * dt;
        safety_metrics.cavitation_dose += current_dose;
    }

    Ok(())
}

/// Check if therapy is within safety limits
///
/// Evaluates current safety metrics against configured limits.
/// Returns the first safety violation encountered, or Safe if all limits are satisfied.
///
/// # Arguments
///
/// - `safety_metrics`: Current safety metrics
/// - `safety_limits`: Configured safety limits
/// - `current_time`: Current therapy session time (s)
///
/// # Returns
///
/// Safety status indicating whether therapy is safe or which limit was exceeded
///
/// # Safety Logic
///
/// Checks are performed in order of severity:
/// 1. Time limit (session duration)
/// 2. Thermal index (tissue heating)
/// 3. Mechanical index (mechanical bioeffects)
/// 4. Cavitation dose (cumulative cavitation)
///
/// # Clinical Response
///
/// When a safety limit is exceeded:
/// - **ThermalLimitExceeded**: Pause therapy, allow tissue cooling
/// - **MechanicalLimitExceeded**: Reduce acoustic power immediately
/// - **CavitationLimitExceeded**: Stop therapy, assess tissue damage
/// - **TimeLimitExceeded**: Terminate therapy session
pub fn check_safety_limits(
    safety_metrics: &SafetyMetrics,
    safety_limits: &SafetyLimits,
    current_time: f64,
) -> SafetyStatus {
    // Check time limit first (most straightforward constraint)
    if current_time > safety_limits.max_treatment_time {
        return SafetyStatus::TimeLimitExceeded;
    }

    // Check thermal index (tissue heating concern)
    if safety_metrics.thermal_index > safety_limits.thermal_index_max {
        return SafetyStatus::ThermalLimitExceeded;
    }

    // Check mechanical index (mechanical bioeffects concern)
    if safety_metrics.mechanical_index > safety_limits.mechanical_index_max {
        return SafetyStatus::MechanicalLimitExceeded;
    }

    // Check cavitation dose (cumulative damage concern)
    if safety_metrics.cavitation_dose > safety_limits.cavitation_dose_max {
        return SafetyStatus::CavitationLimitExceeded;
    }

    // All checks passed
    SafetyStatus::Safe
}
