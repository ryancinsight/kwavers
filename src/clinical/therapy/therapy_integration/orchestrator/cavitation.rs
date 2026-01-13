//! Cavitation Control for Histotripsy and Oncotripsy
//!
//! This module provides cavitation detection and control for cavitation-based therapy modalities
//! including histotripsy and oncotripsy. It implements feedback control algorithms to maintain
//! desired cavitation levels for effective tissue ablation while ensuring safety.
//!
//! ## Cavitation-Based Therapy
//!
//! - **Histotripsy**: Mechanical tissue ablation using controlled cavitation bubble clouds
//! - **Oncotripsy**: Tumor-specific histotripsy with enhanced selectivity
//!
//! ## Control Strategy
//!
//! - Real-time cavitation detection from acoustic signals
//! - Feedback control to maintain target cavitation intensity
//! - Adaptive algorithms for tissue heterogeneity
//! - Safety monitoring and automatic shutdown
//!
//! ## References
//!
//! - Hall et al. (2010): "Histotripsy: minimally invasive tissue ablation using cavitation"
//! - Xu et al. (2016): "Oncotripsy: targeted cancer therapy using tumor-specific cavitation"
//! - Maxwell et al. (2013): "Cavitation clouds in tissue: replication and translation"

use crate::core::error::KwaversResult;
use crate::physics::cavitation_control::FeedbackController;
use ndarray::Array3;

use super::super::config::AcousticTherapyParams;
use super::super::state::AcousticField;

/// Update cavitation control
///
/// Processes acoustic field through feedback controller to detect and control cavitation.
/// Uses pressure amplitude and control feedback to determine spatial cavitation activity.
///
/// # Arguments
///
/// - `cavitation_controller`: Feedback controller for cavitation monitoring and control
/// - `acoustic_field`: Current acoustic field
/// - `acoustic_params`: Therapy acoustic parameters
/// - `dt`: Time step (s)
///
/// # Returns
///
/// 3D cavitation activity map (0-1 normalized)
///
/// # Control Algorithm
///
/// 1. **Signal Processing**: Extract pressure amplitude from acoustic field
/// 2. **Detection**: Identify regions exceeding cavitation threshold
/// 3. **Feedback Control**: Calculate control output based on target intensity
/// 4. **Activity Mapping**: Map pressure and control output to spatial cavitation activity
///
/// ## Cavitation Threshold
///
/// Threshold set at 10% of peak negative pressure (PNP). Regions above this threshold
/// are considered to have active cavitation potential.
///
/// ## Activity Calculation
///
/// Activity level scales with:
/// - Pressure amplitude relative to threshold
/// - Control feedback intensity
/// - Clamped to [0, 1] range
///
/// # References
///
/// - Apfel & Holland (1991): "Gauging the likelihood of cavitation"
/// - Maxwell et al. (2013): "Cavitation clouds in tissue: replication and translation"
pub fn update_cavitation_control(
    cavitation_controller: &mut FeedbackController,
    acoustic_field: &AcousticField,
    acoustic_params: &AcousticTherapyParams,
    _dt: f64,
) -> KwaversResult<Array3<f64>> {
    // Process the acoustic signal through the feedback controller
    // Use pressure field as the input signal for cavitation detection and control
    let signal = acoustic_field.pressure.as_slice().unwrap();
    let array_view = ndarray::ArrayView1::from(signal);
    let control_output = cavitation_controller.process(&array_view);

    // Extract cavitation activity using detector-based approach
    // Use the control output to determine cavitation activity levels
    let mut cavitation_activity = Array3::zeros(acoustic_field.pressure.dim());

    // Map control output to cavitation activity based on detected intensity
    // High intensity indicates active cavitation, use threshold-based mapping
    let cavitation_threshold = acoustic_params.pnp * 0.1; // 10% of peak pressure

    for ((i, j, k), pressure_val) in acoustic_field.pressure.indexed_iter() {
        // Calculate cavitation activity based on pressure amplitude and control feedback
        let pressure_amplitude = pressure_val.abs();
        if pressure_amplitude > cavitation_threshold {
            // Scale activity based on pressure relative to threshold and control intensity
            let activity_level = ((pressure_amplitude - cavitation_threshold)
                / (acoustic_params.pnp - cavitation_threshold))
                .clamp(0.0, 1.0);
            cavitation_activity[[i, j, k]] = activity_level * control_output.cavitation_intensity;
        }
    }

    Ok(cavitation_activity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cavitation_control::{ControlStrategy, FeedbackConfig};

    #[test]
    fn test_cavitation_control_no_activity_below_threshold() {
        // Create feedback controller
        let config = FeedbackConfig {
            strategy: ControlStrategy::AmplitudeOnly,
            target_intensity: 0.8,
            max_amplitude: 1.0,
            min_amplitude: 0.0,
            response_time: 0.001,
            safety_factor: 0.5,
            enable_adaptive: true,
        };
        let mut controller = FeedbackController::new(config, 1e6, 1000.0);

        // Create acoustic field with pressure below cavitation threshold
        let acoustic_field = AcousticField {
            pressure: Array3::from_elem((8, 8, 8), 5000.0), // 5 kPa << threshold
            velocity_x: Array3::zeros((8, 8, 8)),
            velocity_y: Array3::zeros((8, 8, 8)),
            velocity_z: Array3::zeros((8, 8, 8)),
        };

        let acoustic_params = AcousticTherapyParams {
            frequency: 1e6,
            pnp: 1e6, // 1 MPa, threshold = 0.1 MPa
            prf: 100.0,
            duty_cycle: 0.01,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        };

        // Update cavitation control
        let activity =
            update_cavitation_control(&mut controller, &acoustic_field, &acoustic_params, 0.001)
                .unwrap();

        // Activity should be zero (pressure below threshold)
        assert!(activity.iter().all(|&a| a == 0.0));
    }

    #[test]
    fn test_cavitation_control_activity_above_threshold() {
        // Create feedback controller
        let config = FeedbackConfig {
            strategy: ControlStrategy::AmplitudeOnly,
            target_intensity: 0.8,
            max_amplitude: 1.0,
            min_amplitude: 0.0,
            response_time: 0.001,
            safety_factor: 0.5,
            enable_adaptive: true,
        };
        let mut controller = FeedbackController::new(config, 1e6, 1000.0);

        // Create acoustic field with pressure above cavitation threshold
        let acoustic_field = AcousticField {
            pressure: Array3::from_elem((8, 8, 8), 5e5), // 0.5 MPa > threshold (0.1 MPa)
            velocity_x: Array3::zeros((8, 8, 8)),
            velocity_y: Array3::zeros((8, 8, 8)),
            velocity_z: Array3::zeros((8, 8, 8)),
        };

        let acoustic_params = AcousticTherapyParams {
            frequency: 1e6,
            pnp: 1e6, // 1 MPa, threshold = 0.1 MPa
            prf: 100.0,
            duty_cycle: 0.01,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        };

        // Update cavitation control
        let activity =
            update_cavitation_control(&mut controller, &acoustic_field, &acoustic_params, 0.001)
                .unwrap();

        // Activity should be non-zero (pressure above threshold)
        assert!(activity.iter().any(|&a| a > 0.0));

        // Activity should be bounded [0, 1]
        assert!(activity.iter().all(|&a| a >= 0.0 && a <= 1.0));
    }

    #[test]
    fn test_cavitation_control_spatial_variation() {
        // Create feedback controller
        let config = FeedbackConfig {
            strategy: ControlStrategy::AmplitudeOnly,
            target_intensity: 0.6,
            max_amplitude: 1.0,
            min_amplitude: 0.0,
            response_time: 0.002,
            safety_factor: 0.7,
            enable_adaptive: true,
        };
        let mut controller = FeedbackController::new(config, 1e6, 1000.0);

        // Create acoustic field with spatial variation
        let mut pressure = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    // Higher pressure at center, lower at edges
                    let dist_from_center = ((i as f64 - 5.0).powi(2)
                        + (j as f64 - 5.0).powi(2)
                        + (k as f64 - 5.0).powi(2))
                    .sqrt();
                    pressure[[i, j, k]] = 8e5 * (-dist_from_center / 5.0).exp();
                    // Max 0.8 MPa
                }
            }
        }

        let acoustic_field = AcousticField {
            pressure,
            velocity_x: Array3::zeros((10, 10, 10)),
            velocity_y: Array3::zeros((10, 10, 10)),
            velocity_z: Array3::zeros((10, 10, 10)),
        };

        let acoustic_params = AcousticTherapyParams {
            frequency: 1e6,
            pnp: 1e6,
            prf: 100.0,
            duty_cycle: 0.01,
            focal_depth: 0.05,
            treatment_volume: 1.0,
        };

        // Update cavitation control
        let activity =
            update_cavitation_control(&mut controller, &acoustic_field, &acoustic_params, 0.001)
                .unwrap();

        // Center should have higher activity than edges
        let center_activity = activity[(5, 5, 5)];
        let edge_activity = activity[(0, 0, 0)];
        assert!(center_activity >= edge_activity);

        // All activity values should be in valid range
        assert!(activity.iter().all(|&a| a >= 0.0 && a <= 1.0));
    }
}
