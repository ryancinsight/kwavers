//! Feedback controller for cavitation control
//!
//! This module provides a unified interface to the modular control components
//! in the control/ subdirectory.

use super::control::{AdaptiveController, SafetyLimits, SafetyMonitor, StateEstimator};
// Import types that will be re-exported
pub use super::control::{ControlOutput, ControlStrategy, FeedbackConfig};
pub use super::detection::CavitationMetrics;
use super::detection::{CavitationDetector, SpectralDetector};
use super::pid_controller::{PIDConfig, PIDController, PIDGains};
use ndarray::ArrayView1;

/// Main feedback controller
pub struct FeedbackController {
    config: FeedbackConfig,
    detector: Box<dyn CavitationDetector>,
    pid_controller: PIDController,
    state_estimator: StateEstimator,
    safety_monitor: SafetyMonitor,
    adaptive_controller: AdaptiveController,
}

impl std::fmt::Debug for FeedbackController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedbackController")
            .field("config", &self.config)
            .field("detector", &"<dyn CavitationDetector>")
            .field("pid_controller", &self.pid_controller)
            .finish()
    }
}

impl FeedbackController {
    /// Create new feedback controller
    pub fn new(config: FeedbackConfig, fundamental_freq: f64, sample_rate: f64) -> Self {
        // Initialize PID controller
        let pid_config = PIDConfig {
            gains: PIDGains {
                kp: 1.0,
                ki: 0.1,
                kd: 0.01,
            },
            sample_time: 1.0 / sample_rate,
            output_min: config.min_amplitude,
            output_max: config.max_amplitude,
            ..Default::default()
        };

        let mut pid_controller = PIDController::new(pid_config);
        pid_controller.set_setpoint(config.target_intensity);

        Self {
            config: config.clone(),
            detector: Box::new(SpectralDetector::new(fundamental_freq, sample_rate)),
            pid_controller,
            state_estimator: StateEstimator::new(),
            safety_monitor: SafetyMonitor::new(SafetyLimits::default()),
            adaptive_controller: AdaptiveController::new(0.01),
        }
    }

    /// Process control loop
    pub fn process(&mut self, signal: &ArrayView1<f64>) -> ControlOutput {
        // Detect cavitation
        let raw_metrics = self.detector.detect(signal);

        // Estimate state with temporal smoothing
        let metrics = self.state_estimator.estimate(&raw_metrics);

        // Check safety
        let safe = self.safety_monitor.check_safety(&metrics);

        // Calculate control error
        let error = self.config.target_intensity - metrics.intensity;

        // Get PID control output
        let pid_output = self.pid_controller.update(metrics.intensity);

        // Apply control strategy
        let mut output = match self.config.strategy {
            ControlStrategy::AmplitudeOnly => ControlOutput {
                amplitude: pid_output.control_signal,
                frequency: 0.0,
                pulse_width: 1.0,
                phase: 0.0,
                modulation_index: 0.0,
                safety_limited: false,
                error,
                cavitation_intensity: metrics.intensity,
                control_active: safe,
            },
            ControlStrategy::PulseWidthModulation => {
                // Use duty cycle based on intensity
                let duty_cycle = (1.0 - error / self.config.target_intensity).clamp(0.0, 1.0);
                ControlOutput {
                    amplitude: self.config.max_amplitude,
                    frequency: 0.0,
                    pulse_width: duty_cycle,
                    phase: 0.0,
                    modulation_index: duty_cycle,
                    safety_limited: false,
                    error,
                    cavitation_intensity: metrics.intensity,
                    control_active: safe,
                }
            }
            ControlStrategy::FrequencyModulation => {
                let freq_shift = pid_output.control_signal * 0.1; // ±10% frequency
                ControlOutput {
                    amplitude: self.config.max_amplitude,
                    frequency: freq_shift,
                    pulse_width: 1.0,
                    phase: 0.0,
                    modulation_index: freq_shift,
                    safety_limited: false,
                    error,
                    cavitation_intensity: metrics.intensity,
                    control_active: safe,
                }
            }
            ControlStrategy::Combined => {
                // Combine amplitude and PWM
                let duty_cycle = (1.0 - error / self.config.target_intensity).clamp(0.0, 1.0);
                ControlOutput {
                    amplitude: pid_output.control_signal,
                    frequency: 0.0,
                    pulse_width: duty_cycle,
                    phase: 0.0,
                    modulation_index: duty_cycle,
                    safety_limited: false,
                    error,
                    cavitation_intensity: metrics.intensity,
                    control_active: safe,
                }
            }
        };

        // Apply safety limits
        output = self.safety_monitor.apply_safety_limits(output);

        // Adapt parameters if enabled
        let target = self.config.target_intensity;
        self.adaptive_controller
            .adapt_parameters(&mut self.config, &metrics, target);

        output
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.detector.reset();
        self.pid_controller.reset();
        self.state_estimator.reset();
        self.safety_monitor.reset();
        self.adaptive_controller.reset();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: FeedbackConfig) {
        self.config = config;
        self.pid_controller
            .set_setpoint(self.config.target_intensity);
    }

    /// Get current configuration
    pub fn config(&self) -> &FeedbackConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_feedback_controller_creation() {
        let config = FeedbackConfig::default();
        let controller = FeedbackController::new(config, 1e6, 10e6);
        assert_eq!(controller.config().target_intensity, 0.5);
    }

    #[test]
    fn test_control_output() {
        let config = FeedbackConfig::default();
        let mut controller = FeedbackController::new(config, 1e6, 10e6);
        let signal = Array1::zeros(1024);
        let output = controller.process(&signal.view());
        assert!(output.amplitude >= 0.0);
        assert!(output.amplitude <= 1.0);
    }
}
