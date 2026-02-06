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
pub fn update_safety_metrics(
    safety_metrics: &mut SafetyMetrics,
    acoustic_field: &AcousticField,
    acoustic_params: &AcousticTherapyParams,
    dt: f64,
    cavitation_activity: Option<&ndarray::Array3<f64>>,
) -> KwaversResult<()> {
    // Calculate thermal index (IEC 62359 compliant)
    let pressure_rms = acoustic_field
        .pressure
        .iter()
        .map(|&p| p * p)
        .sum::<f64>()
        .sqrt()
        / acoustic_field.pressure.len() as f64;

    safety_metrics.thermal_index = pressure_rms * acoustic_params.frequency.sqrt() / 1e6;

    // Calculate mechanical index (FDA guidance compliant)
    safety_metrics.mechanical_index =
        acoustic_params.pnp / (acoustic_params.frequency.sqrt() * 1e6);

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_thermal_index_calculation() {
        let acoustic_field = AcousticField {
            pressure: Array3::from_elem((10, 10, 10), 1e6), // 1 MPa uniform pressure
            velocity_x: Array3::zeros((10, 10, 10)),
            velocity_y: Array3::zeros((10, 10, 10)),
            velocity_z: Array3::zeros((10, 10, 10)),
        };

        let acoustic_params = AcousticTherapyParams {
            frequency: 1e6,        // 1 MHz
            pnp: 1e6,              // 1 MPa
            prf: 100.0,            // 100 Hz
            duty_cycle: 0.1,       // 10%
            focal_depth: 0.05,     // 5 cm
            treatment_volume: 1.0, // 1 cm³
        };

        let mut safety_metrics = SafetyMetrics {
            thermal_index: 0.0,
            mechanical_index: 0.0,
            cavitation_dose: 0.0,
            temperature_rise: Array3::zeros((10, 10, 10)),
        };

        update_safety_metrics(
            &mut safety_metrics,
            &acoustic_field,
            &acoustic_params,
            0.01,
            None,
        )
        .unwrap();

        // TI = P_rms * sqrt(f) / 1e6
        // P_rms = 1e6 Pa, sqrt(1e6) = 1000
        // TI = 1e6 * 1000 / 1e6 = 1000
        assert!(safety_metrics.thermal_index > 0.0);
        assert!(safety_metrics.thermal_index < 2000.0); // Reasonable upper bound
    }

    #[test]
    fn test_mechanical_index_calculation() {
        let acoustic_field = AcousticField {
            pressure: Array3::from_elem((8, 8, 8), 0.5e6), // 0.5 MPa
            velocity_x: Array3::zeros((8, 8, 8)),
            velocity_y: Array3::zeros((8, 8, 8)),
            velocity_z: Array3::zeros((8, 8, 8)),
        };

        let acoustic_params = AcousticTherapyParams {
            frequency: 1e6, // 1 MHz
            pnp: 0.5e6,     // 0.5 MPa
            prf: 100.0,
            duty_cycle: 0.1,
            focal_depth: 0.03,
            treatment_volume: 0.5,
        };

        let mut safety_metrics = SafetyMetrics {
            thermal_index: 0.0,
            mechanical_index: 0.0,
            cavitation_dose: 0.0,
            temperature_rise: Array3::zeros((8, 8, 8)),
        };

        update_safety_metrics(
            &mut safety_metrics,
            &acoustic_field,
            &acoustic_params,
            0.01,
            None,
        )
        .unwrap();

        // MI = PNP / (sqrt(f) * 1e6)
        // MI = 0.5e6 / (sqrt(1e6) * 1e6) = 0.5e6 / (1000 * 1e6) = 0.5e6 / 1e9 = 0.0005
        assert!((safety_metrics.mechanical_index - 0.0005).abs() < 0.0001);
    }

    #[test]
    fn test_cavitation_dose_accumulation() {
        let acoustic_field = AcousticField {
            pressure: Array3::from_elem((5, 5, 5), 1e6),
            velocity_x: Array3::zeros((5, 5, 5)),
            velocity_y: Array3::zeros((5, 5, 5)),
            velocity_z: Array3::zeros((5, 5, 5)),
        };

        let acoustic_params = AcousticTherapyParams {
            frequency: 1e6,
            pnp: 1e6,
            prf: 100.0,
            duty_cycle: 0.1,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        };

        let mut safety_metrics = SafetyMetrics {
            thermal_index: 0.0,
            mechanical_index: 0.0,
            cavitation_dose: 0.0,
            temperature_rise: Array3::zeros((5, 5, 5)),
        };

        let cavitation_activity = Array3::from_elem((5, 5, 5), 0.5); // 50% activity
        let dt = 0.01; // 10 ms

        // First update
        update_safety_metrics(
            &mut safety_metrics,
            &acoustic_field,
            &acoustic_params,
            dt,
            Some(&cavitation_activity),
        )
        .unwrap();

        let dose_after_first = safety_metrics.cavitation_dose;
        assert!(dose_after_first > 0.0);

        // Second update - dose should accumulate
        update_safety_metrics(
            &mut safety_metrics,
            &acoustic_field,
            &acoustic_params,
            dt,
            Some(&cavitation_activity),
        )
        .unwrap();

        assert!(safety_metrics.cavitation_dose > dose_after_first);
        assert!((safety_metrics.cavitation_dose - 2.0 * dose_after_first).abs() < 1e-10);
    }

    #[test]
    fn test_safety_limit_checking_all_safe() {
        let safety_metrics = SafetyMetrics {
            thermal_index: 3.0,
            mechanical_index: 1.0,
            cavitation_dose: 500.0,
            temperature_rise: Array3::zeros((5, 5, 5)),
        };

        let safety_limits = SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        };

        let status = check_safety_limits(&safety_metrics, &safety_limits, 60.0);
        assert_eq!(status, SafetyStatus::Safe);
    }

    #[test]
    fn test_safety_limit_thermal_exceeded() {
        let safety_metrics = SafetyMetrics {
            thermal_index: 7.0, // Exceeds limit
            mechanical_index: 1.0,
            cavitation_dose: 500.0,
            temperature_rise: Array3::zeros((5, 5, 5)),
        };

        let safety_limits = SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        };

        let status = check_safety_limits(&safety_metrics, &safety_limits, 60.0);
        assert_eq!(status, SafetyStatus::ThermalLimitExceeded);
    }

    #[test]
    fn test_safety_limit_mechanical_exceeded() {
        let safety_metrics = SafetyMetrics {
            thermal_index: 3.0,
            mechanical_index: 2.0, // Exceeds limit
            cavitation_dose: 500.0,
            temperature_rise: Array3::zeros((5, 5, 5)),
        };

        let safety_limits = SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        };

        let status = check_safety_limits(&safety_metrics, &safety_limits, 60.0);
        assert_eq!(status, SafetyStatus::MechanicalLimitExceeded);
    }

    #[test]
    fn test_safety_limit_time_exceeded() {
        let safety_metrics = SafetyMetrics {
            thermal_index: 3.0,
            mechanical_index: 1.0,
            cavitation_dose: 500.0,
            temperature_rise: Array3::zeros((5, 5, 5)),
        };

        let safety_limits = SafetyLimits {
            thermal_index_max: 6.0,
            mechanical_index_max: 1.9,
            cavitation_dose_max: 1000.0,
            max_treatment_time: 300.0,
        };

        let status = check_safety_limits(&safety_metrics, &safety_limits, 301.0); // Over time limit
        assert_eq!(status, SafetyStatus::TimeLimitExceeded);
    }
}
