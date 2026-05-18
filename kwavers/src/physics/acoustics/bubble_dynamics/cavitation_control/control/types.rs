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
pub struct CavitationSafetyLimits {
    pub max_intensity: f64,
    pub max_temperature: f64,
    pub max_pressure: f64,
    pub emergency_stop_threshold: f64,
}

impl Default for CavitationSafetyLimits {
    fn default() -> Self {
        Self {
            max_intensity: 0.9,
            max_temperature: 43.0, // °C
            max_pressure: 10e6,    // Pa
            emergency_stop_threshold: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FeedbackConfig::default satisfies amplitude ordering: min_amplitude < max_amplitude.
    #[test]
    fn default_feedback_config_amplitude_ordering() {
        let cfg = FeedbackConfig::default();
        assert!(
            cfg.min_amplitude < cfg.max_amplitude,
            "min_amplitude ({}) must be < max_amplitude ({})",
            cfg.min_amplitude,
            cfg.max_amplitude
        );
    }

    /// CavitationSafetyLimits::default satisfies intensity ordering: max_intensity < emergency threshold.
    #[test]
    fn default_safety_limits_intensity_ordering() {
        let lim = CavitationSafetyLimits::default();
        assert!(
            lim.max_intensity < lim.emergency_stop_threshold,
            "max_intensity ({}) must be < emergency_stop_threshold ({})",
            lim.max_intensity,
            lim.emergency_stop_threshold
        );
        assert!(
            lim.max_temperature > 37.0, // above body temperature
            "max temperature ({}) must be above 37°C",
            lim.max_temperature
        );
    }

    /// ControlStrategy variants are pairwise distinct.
    #[test]
    fn control_strategy_variants_distinct() {
        assert_ne!(
            ControlStrategy::AmplitudeOnly,
            ControlStrategy::PulseWidthModulation
        );
        assert_ne!(
            ControlStrategy::FrequencyModulation,
            ControlStrategy::Combined
        );
    }

    /// Clone produces equal copy of FeedbackConfig.
    #[test]
    fn feedback_config_clone_equal() {
        let cfg = FeedbackConfig::default();
        let c = cfg.clone();
        assert!((cfg.target_intensity - c.target_intensity).abs() < 1e-15);
        assert_eq!(cfg.strategy, c.strategy);
    }
}
