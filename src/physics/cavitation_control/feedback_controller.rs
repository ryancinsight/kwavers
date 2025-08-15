//! Feedback Controller for Cavitation Control
//! 
//! Implements a complete negative feedback control system for maintaining
//! desired cavitation levels using real-time monitoring and control.
//! 
//! References:
//! - Hockham et al. (2013): "Real-time control system for sustaining thermally relevant acoustic cavitation"
//! - Arvanitis et al. (2013): "Cavitation-enhanced nonthermal ablation in deep brain targets"
//! - Sun et al. (2017): "Closed-loop control of targeted ultrasound drug delivery"

use crate::physics::cavitation_control::{
    pid_controller::{PIDController, PIDConfig, PIDGains, ControllerOutput},
    power_modulation::{PowerModulator, ModulationScheme, PowerControl, AmplitudeController, DutyCycleController},
    cavitation_detector::{CavitationDetector, SpectralDetector, CavitationState},
};
use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;

// Feedback control constants
/// Default target cavitation intensity (0-1)
const DEFAULT_TARGET_INTENSITY: f64 = 0.5;

/// Control loop update rate (Hz)
const CONTROL_UPDATE_RATE: f64 = 100.0;

/// Minimum control update period (seconds)
const MIN_UPDATE_PERIOD: f64 = 0.001;

/// Maximum control history length
const MAX_HISTORY_LENGTH: usize = 1000;

/// Control dead zone to prevent oscillations
const CONTROL_DEAD_ZONE: f64 = 0.02;

/// Safety shutdown threshold
const SAFETY_SHUTDOWN_THRESHOLD: f64 = 0.95;

/// Control strategies for feedback
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlStrategy {
    AmplitudeOnly,      // Control only amplitude
    DutyCycleOnly,      // Control only duty cycle
    Combined,           // Control both amplitude and duty cycle
    Cascaded,           // Cascaded control (coarse/fine)
    Predictive,         // Model predictive control
}

/// Feedback configuration
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    pub strategy: ControlStrategy,
    pub target_intensity: f64,
    pub pid_gains: PIDGains,
    pub update_rate: f64,
    pub safety_enabled: bool,
    pub dead_zone: f64,
    pub max_amplitude: f64,
    pub max_duty_cycle: f64,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            strategy: ControlStrategy::Combined,
            target_intensity: DEFAULT_TARGET_INTENSITY,
            pid_gains: PIDGains::default(),
            update_rate: CONTROL_UPDATE_RATE,
            safety_enabled: true,
            dead_zone: CONTROL_DEAD_ZONE,
            max_amplitude: 1.0,
            max_duty_cycle: 0.8,
        }
    }
}

/// Control output from feedback system
#[derive(Debug, Clone)]
pub struct ControlOutput {
    pub amplitude: f64,
    pub duty_cycle: f64,
    pub modulation_scheme: ModulationScheme,
    pub error: f64,
    pub cavitation_intensity: f64,
    pub control_active: bool,
}

/// Cavitation metrics for control
pub use crate::physics::cavitation_control::cavitation_detector::CavitationMetrics;

/// Main feedback controller
pub struct FeedbackController {
    config: FeedbackConfig,
    pid_controller: PIDController,
    amplitude_controller: AmplitudeController,
    duty_cycle_controller: DutyCycleController,
    power_modulator: PowerModulator,
    cavitation_detector: Box<dyn CavitationDetector>,
    control_history: VecDeque<ControlOutput>,
    time_since_update: f64,
    emergency_stop: bool,
    state_estimator: StateEstimator,
}

impl FeedbackController {
    pub fn new(
        config: FeedbackConfig,
        fundamental_freq: f64,
        sample_rate: f64,
    ) -> Self {
        let pid_config = PIDConfig {
            gains: config.pid_gains,
            sample_time: 1.0 / config.update_rate,
            output_min: -1.0,
            output_max: 1.0,
            ..Default::default()
        };
        
        let mut pid_controller = PIDController::new(pid_config);
        pid_controller.set_setpoint(config.target_intensity);
        
        Self {
            config: config.clone(),
            pid_controller,
            amplitude_controller: AmplitudeController::new(config.max_amplitude),
            duty_cycle_controller: DutyCycleController::new(config.max_duty_cycle),
            power_modulator: PowerModulator::new(ModulationScheme::DutyCycleControl, sample_rate),
            cavitation_detector: Box::new(SpectralDetector::new(fundamental_freq, sample_rate)),
            control_history: VecDeque::with_capacity(MAX_HISTORY_LENGTH),
            time_since_update: 0.0,
            emergency_stop: false,
            state_estimator: StateEstimator::new(),
        }
    }
    
    /// Process acoustic signal and update control
    pub fn update(&mut self, acoustic_signal: &ArrayView1<f64>, dt: f64) -> ControlOutput {
        self.time_since_update += dt;
        
        // Check if it's time to update control
        if self.time_since_update < (1.0 / self.config.update_rate).max(MIN_UPDATE_PERIOD) {
            return self.get_current_output();
        }
        
        self.time_since_update = 0.0;
        
        // Detect cavitation
        let mut metrics = self.cavitation_detector.detect(acoustic_signal);
        
        // Apply state estimation for noise reduction
        metrics = self.state_estimator.estimate(metrics);
        
        // Check safety conditions
        if self.config.safety_enabled {
            self.check_safety(&metrics);
        }
        
        if self.emergency_stop {
            return self.emergency_shutdown();
        }
        
        // Calculate control error
        let error = self.config.target_intensity - metrics.intensity;
        
        // Apply dead zone to prevent oscillations
        let effective_error = if error.abs() < self.config.dead_zone {
            0.0
        } else {
            error
        };
        
        // Get PID control output
        let pid_output = self.pid_controller.update(metrics.intensity);
        
        // Apply control strategy
        let (amplitude, duty_cycle) = match self.config.strategy {
            ControlStrategy::AmplitudeOnly => {
                let amp = self.apply_amplitude_control(pid_output.control_signal);
                (amp, self.config.max_duty_cycle)
            }
            
            ControlStrategy::DutyCycleOnly => {
                let duty = self.apply_duty_cycle_control(pid_output.control_signal);
                (self.config.max_amplitude, duty)
            }
            
            ControlStrategy::Combined => {
                self.apply_combined_control(pid_output.control_signal, &metrics)
            }
            
            ControlStrategy::Cascaded => {
                self.apply_cascaded_control(pid_output.control_signal, &metrics)
            }
            
            ControlStrategy::Predictive => {
                self.apply_predictive_control(pid_output.control_signal, &metrics)
            }
        };
        
        // Update power modulator
        self.power_modulator.update_control(PowerControl {
            amplitude,
            duty_cycle,
            ..Default::default()
        });
        
        // Create control output
        let output = ControlOutput {
            amplitude,
            duty_cycle,
            modulation_scheme: self.get_modulation_scheme(&metrics),
            error: effective_error,
            cavitation_intensity: metrics.intensity,
            control_active: !self.emergency_stop,
        };
        
        // Store in history
        self.control_history.push_back(output.clone());
        if self.control_history.len() > MAX_HISTORY_LENGTH {
            self.control_history.pop_front();
        }
        
        output
    }
    
    /// Apply amplitude-only control
    fn apply_amplitude_control(&mut self, control_signal: f64) -> f64 {
        let target = (self.config.max_amplitude * (1.0 + control_signal)).clamp(0.0, self.config.max_amplitude);
        self.amplitude_controller.set_target(target);
        self.amplitude_controller.update(1.0 / self.config.update_rate)
    }
    
    /// Apply duty cycle-only control
    fn apply_duty_cycle_control(&mut self, control_signal: f64) -> f64 {
        let target = (self.config.max_duty_cycle * (1.0 + control_signal)).clamp(0.0, self.config.max_duty_cycle);
        self.duty_cycle_controller.set_target(target);
        self.duty_cycle_controller.update(1.0 / self.config.update_rate)
    }
    
    /// Apply combined amplitude and duty cycle control
    fn apply_combined_control(&mut self, control_signal: f64, metrics: &CavitationMetrics) -> (f64, f64) {
        // Use amplitude for fine control, duty cycle for coarse control
        let amplitude_contribution = 0.7;
        let duty_cycle_contribution = 0.3;
        
        let amp_signal = control_signal * amplitude_contribution;
        let duty_signal = control_signal * duty_cycle_contribution;
        
        // Prefer amplitude control when stable cavitation is detected
        let (amp_weight, duty_weight) = match metrics.state {
            CavitationState::Stable => (0.8, 0.2),
            CavitationState::Inertial => (0.3, 0.7),
            _ => (0.5, 0.5),
        };
        
        let amplitude = self.apply_amplitude_control(amp_signal * amp_weight);
        let duty_cycle = self.apply_duty_cycle_control(duty_signal * duty_weight);
        
        (amplitude, duty_cycle)
    }
    
    /// Apply cascaded control (coarse/fine)
    fn apply_cascaded_control(&mut self, control_signal: f64, metrics: &CavitationMetrics) -> (f64, f64) {
        // Use duty cycle for coarse control, amplitude for fine control
        let threshold = 0.2;
        
        if control_signal.abs() > threshold {
            // Large error: adjust duty cycle
            let duty_cycle = self.apply_duty_cycle_control(control_signal);
            (self.amplitude_controller.amplitude(), duty_cycle)
        } else {
            // Small error: adjust amplitude
            let amplitude = self.apply_amplitude_control(control_signal);
            (amplitude, self.duty_cycle_controller.duty_cycle())
        }
    }
    
    /// Apply model predictive control
    fn apply_predictive_control(&mut self, control_signal: f64, metrics: &CavitationMetrics) -> (f64, f64) {
        // Simple predictive control based on trend
        let trend = self.state_estimator.get_trend();
        let predicted_signal = control_signal + trend * 0.1;
        
        // Apply control with prediction
        self.apply_combined_control(predicted_signal, metrics)
    }
    
    /// Check safety conditions
    fn check_safety(&mut self, metrics: &CavitationMetrics) {
        // Check for excessive cavitation
        if metrics.intensity > SAFETY_SHUTDOWN_THRESHOLD {
            self.emergency_stop = true;
            log::warn!("Emergency stop: Excessive cavitation detected");
        }
        
        // Check for inertial cavitation in sensitive applications
        if metrics.state == CavitationState::Inertial && metrics.confidence > 0.8 {
            // Reduce power immediately
            self.amplitude_controller.set_target(self.config.max_amplitude * 0.5);
            self.duty_cycle_controller.set_target(self.config.max_duty_cycle * 0.5);
        }
    }
    
    /// Emergency shutdown
    fn emergency_shutdown(&mut self) -> ControlOutput {
        self.amplitude_controller.set_target(0.0);
        self.duty_cycle_controller.set_target(MIN_DUTY_CYCLE);
        
        ControlOutput {
            amplitude: 0.0,
            duty_cycle: MIN_DUTY_CYCLE,
            modulation_scheme: ModulationScheme::Pulsed,
            error: 0.0,
            cavitation_intensity: 0.0,
            control_active: false,
        }
    }
    
    /// Get appropriate modulation scheme based on cavitation state
    fn get_modulation_scheme(&self, metrics: &CavitationMetrics) -> ModulationScheme {
        match metrics.state {
            CavitationState::None => ModulationScheme::Continuous,
            CavitationState::Stable => ModulationScheme::AmplitudeModulation,
            CavitationState::Inertial => ModulationScheme::DutyCycleControl,
            CavitationState::Transient => ModulationScheme::Ramped,
        }
    }
    
    /// Get current control output
    fn get_current_output(&self) -> ControlOutput {
        self.control_history.back().cloned().unwrap_or(ControlOutput {
            amplitude: self.amplitude_controller.amplitude(),
            duty_cycle: self.duty_cycle_controller.duty_cycle(),
            modulation_scheme: ModulationScheme::Continuous,
            error: 0.0,
            cavitation_intensity: 0.0,
            control_active: !self.emergency_stop,
        })
    }
    
    /// Reset controller
    pub fn reset(&mut self) {
        self.pid_controller.reset();
        self.control_history.clear();
        self.time_since_update = 0.0;
        self.emergency_stop = false;
        self.cavitation_detector.reset();
        self.state_estimator.reset();
    }
    
    /// Set new target intensity
    pub fn set_target(&mut self, target: f64) {
        self.config.target_intensity = target.clamp(0.0, 1.0);
        self.pid_controller.set_setpoint(self.config.target_intensity);
    }
    
    /// Clear emergency stop
    pub fn clear_emergency_stop(&mut self) {
        self.emergency_stop = false;
    }
}

/// State estimator for noise reduction
struct StateEstimator {
    history: VecDeque<CavitationMetrics>,
    alpha: f64,
}

impl StateEstimator {
    fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(10),
            alpha: 0.7, // Exponential smoothing factor
        }
    }
    
    fn estimate(&mut self, metrics: CavitationMetrics) -> CavitationMetrics {
        if self.history.is_empty() {
            self.history.push_back(metrics.clone());
            return metrics;
        }
        
        let last = self.history.back().unwrap();
        
        // Exponential smoothing
        let smoothed = CavitationMetrics {
            intensity: self.alpha * metrics.intensity + (1.0 - self.alpha) * last.intensity,
            subharmonic_level: self.alpha * metrics.subharmonic_level + (1.0 - self.alpha) * last.subharmonic_level,
            broadband_level: self.alpha * metrics.broadband_level + (1.0 - self.alpha) * last.broadband_level,
            harmonic_content: self.alpha * metrics.harmonic_content + (1.0 - self.alpha) * last.harmonic_content,
            cavitation_dose: metrics.cavitation_dose, // Don't smooth cumulative dose
            confidence: self.alpha * metrics.confidence + (1.0 - self.alpha) * last.confidence,
            state: metrics.state, // Use current state
        };
        
        self.history.push_back(smoothed.clone());
        if self.history.len() > 10 {
            self.history.pop_front();
        }
        
        smoothed
    }
    
    fn get_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        
        let recent: Vec<f64> = self.history.iter()
            .rev()
            .take(5)
            .map(|m| m.intensity)
            .collect();
        
        if recent.len() < 2 {
            return 0.0;
        }
        
        // Simple linear trend
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..recent.len()).map(|i| (i as f64) * (i as f64)).sum();
        
        (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    }
    
    fn reset(&mut self) {
        self.history.clear();
    }
}

// Re-export MIN_DUTY_CYCLE from power_modulation
const MIN_DUTY_CYCLE: f64 = 0.01;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_feedback_controller() {
        let config = FeedbackConfig::default();
        let mut controller = FeedbackController::new(config, 1e6, 10e6);
        
        // Create test signal
        let signal = Array1::zeros(1024);
        
        // Update controller
        let output = controller.update(&signal.view(), 0.01);
        
        assert!(output.amplitude >= 0.0 && output.amplitude <= 1.0);
        assert!(output.duty_cycle >= 0.0 && output.duty_cycle <= 1.0);
    }
    
    #[test]
    fn test_safety_shutdown() {
        let mut config = FeedbackConfig::default();
        config.safety_enabled = true;
        
        let mut controller = FeedbackController::new(config, 1e6, 10e6);
        
        // Create high-intensity signal that should trigger safety
        let signal = Array1::ones(1024) * 10.0;
        
        let output = controller.update(&signal.view(), 0.01);
        
        // Should eventually trigger safety shutdown
        assert!(!output.control_active || output.amplitude < 1.0);
    }
}