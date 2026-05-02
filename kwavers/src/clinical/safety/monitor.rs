use crate::clinical::therapy::parameters::TherapyParameters;
use std::time::{Duration, Instant};

/// Safety levels for clinical ultrasound systems
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SafetyLevel {
    /// Normal operation within all safety limits
    Normal,
    /// Warning - approaching safety limits
    Warning,
    /// Critical - safety limits exceeded, automatic shutdown required
    Critical,
    /// Emergency - immediate system shutdown
    Emergency,
}

/// IEC 60601-2-37 safety limits for therapeutic ultrasound
#[derive(Debug, Clone)]
pub struct SafetyLimits {
    /// Maximum acoustic power (W)
    pub max_power: f64,
    /// Maximum intensity (W/cm²)
    pub max_intensity: f64,
    /// Maximum temperature rise (°C)
    pub max_temperature_rise: f64,
    /// Maximum treatment time per session (seconds)
    pub max_session_time: f64,
    /// Maximum total dose per treatment course (J)
    pub max_total_dose: f64,
    /// Minimum standoff distance (mm)
    pub min_standoff: f64,
    /// Maximum beam non-uniformity ratio
    pub max_bnur: f64,
}

impl Default for SafetyLimits {
    fn default() -> Self {
        Self {
            max_power: 100.0,
            max_intensity: 3.0,
            max_temperature_rise: 5.0,
            max_session_time: 1800.0,
            max_total_dose: 10000.0,
            min_standoff: 5.0,
            max_bnur: 6.0,
        }
    }
}

/// Real-time safety monitor for therapeutic ultrasound
#[derive(Debug)]
pub struct SafetyMonitor {
    limits: SafetyLimits,
    current_state: SafetyLevel,
    violations: Vec<SafetyViolation>,
    monitoring_enabled: bool,
    last_check: Instant,
    check_interval: Duration,
}

impl SafetyMonitor {
    /// Create new safety monitor with 10 Hz check interval.
    pub fn new(limits: SafetyLimits) -> Self {
        let check_interval = Duration::from_millis(100);
        Self {
            limits,
            current_state: SafetyLevel::Normal,
            violations: Vec::new(),
            monitoring_enabled: true,
            last_check: Instant::now() - check_interval,
            check_interval,
        }
    }

    /// Check current therapy parameters against safety limits.
    pub fn check_safety(&mut self, params: &TherapyParameters) -> SafetyLevel {
        if !self.monitoring_enabled {
            return SafetyLevel::Normal;
        }

        let now = Instant::now();
        if now.duration_since(self.last_check) < self.check_interval {
            return self.current_state;
        }

        self.violations.clear();
        let mut new_state = SafetyLevel::Normal;

        // FDA limit: ~3 MPa peak negative pressure for therapeutic ultrasound
        const MAX_PEAK_NEGATIVE_PRESSURE: f64 = 3.0e6;
        if params.peak_negative_pressure > MAX_PEAK_NEGATIVE_PRESSURE {
            self.violations.push(SafetyViolation {
                parameter: "peak_negative_pressure".to_string(),
                measured_value: params.peak_negative_pressure,
                limit_value: MAX_PEAK_NEGATIVE_PRESSURE,
                severity: SafetyLevel::Critical,
                timestamp: Instant::now(),
                message: "Peak negative pressure exceeds maximum safe limit".to_string(),
            });
            new_state = SafetyLevel::Critical;
        }

        if params.mechanical_index > 1.9 {
            self.violations.push(SafetyViolation {
                parameter: "mechanical_index".to_string(),
                measured_value: params.mechanical_index,
                limit_value: 1.9,
                severity: SafetyLevel::Critical,
                timestamp: Instant::now(),
                message: "Mechanical index exceeds FDA safety limits".to_string(),
            });
            new_state = SafetyLevel::Critical;
        }

        if params.treatment_duration > self.limits.max_session_time {
            self.violations.push(SafetyViolation {
                parameter: "treatment_duration".to_string(),
                measured_value: params.treatment_duration,
                limit_value: self.limits.max_session_time,
                severity: SafetyLevel::Critical,
                timestamp: Instant::now(),
                message: "Treatment duration exceeds maximum limit".to_string(),
            });
            new_state = SafetyLevel::Critical;
        }

        if params.frequency < 0.5e6 || params.frequency > 10e6 {
            self.violations.push(SafetyViolation {
                parameter: "frequency".to_string(),
                measured_value: params.frequency,
                limit_value: 5e6,
                severity: SafetyLevel::Warning,
                timestamp: Instant::now(),
                message: "Frequency outside typical therapeutic ultrasound range".to_string(),
            });
            if new_state < SafetyLevel::Warning {
                new_state = SafetyLevel::Warning;
            }
        }

        self.current_state = new_state;
        self.last_check = now;
        new_state
    }

    /// Get current safety state.
    pub fn safety_state(&self) -> SafetyLevel {
        self.current_state
    }

    /// Get list of current safety violations.
    pub fn violations(&self) -> &[SafetyViolation] {
        &self.violations
    }

    /// Enable or disable safety monitoring.
    pub fn set_monitoring_enabled(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
    }

    /// Return `true` if emergency shutdown is required.
    pub fn requires_emergency_shutdown(&self) -> bool {
        self.current_state >= SafetyLevel::Critical
    }
}

/// Safety violation record
#[derive(Debug, Clone)]
pub struct SafetyViolation {
    pub parameter: String,
    pub measured_value: f64,
    pub limit_value: f64,
    pub severity: SafetyLevel,
    pub timestamp: Instant,
    pub message: String,
}
