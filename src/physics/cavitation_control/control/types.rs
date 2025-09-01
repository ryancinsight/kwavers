//! Control system types and enums

/// Control strategy for cavitation feedback
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlStrategy {
    AmplitudeOnly,        // Control only amplitude
    PulseWidthModulation, // Control pulse width
    FrequencyModulation,  // Control frequency
    Combined,             // Combined control
}

/// Feedback controller configuration
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    pub strategy: ControlStrategy,
    pub target_intensity: f64,
    pub max_amplitude: f64,
    pub min_amplitude: f64,
    pub response_time: f64,
    pub safety_factor: f64,
    pub enable_adaptive: bool,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            strategy: ControlStrategy::AmplitudeOnly,
            target_intensity: 0.5,
            max_amplitude: 1.0,
            min_amplitude: 0.0,
            response_time: 0.1,
            safety_factor: 0.8,
            enable_adaptive: false,
        }
    }
}

/// Control output from feedback controller
#[derive(Debug, Clone)]
pub struct ControlOutput {
    pub amplitude: f64,
    pub frequency: f64,
    pub pulse_width: f64,
    pub phase: f64,
    pub modulation_index: f64,
    pub safety_limited: bool,
    pub error: f64,
    pub cavitation_intensity: f64,
    pub control_active: bool,
}

/// Safety limits for control system
#[derive(Debug, Clone)]
pub struct SafetyLimits {
    pub max_intensity: f64,
    pub max_temperature: f64,
    pub max_pressure: f64,
    pub emergency_stop_threshold: f64,
}

impl Default for SafetyLimits {
    fn default() -> Self {
        Self {
            max_intensity: 0.9,
            max_temperature: 43.0, // °C
            max_pressure: 10e6,    // Pa
            emergency_stop_threshold: 0.95,
        }
    }
}
