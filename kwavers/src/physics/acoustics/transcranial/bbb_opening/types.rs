use ndarray::Array3;

/// Permeability enhancement data
#[derive(Debug, Clone)]
pub struct PermeabilityEnhancement {
    /// Local permeability increase factor
    pub permeability_factor: Array3<f64>,
    /// Opening duration (seconds)
    pub opening_duration: Array3<f64>,
    /// BBB recovery time (hours)
    pub recovery_time: Array3<f64>,
    /// Microbubble concentration effect
    pub microbubble_effect: Array3<f64>,
}

#[derive(Debug, Clone)]
pub struct BBBParameters {
    /// Acoustic frequency (Hz)
    pub frequency: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Duty cycle (%)
    pub duty_cycle: f64,
    /// Treatment duration (s)
    pub duration: f64,
    /// Mechanical index target
    pub target_mi: f64,
    /// Microbubble size distribution (mean radius, std dev) in μm
    pub bubble_size: (f64, f64),
}

impl Default for BBBParameters {
    fn default() -> Self {
        Self {
            frequency: 1.0e6,        // 1 MHz
            prf: 1.0,                // 1 Hz
            duty_cycle: 10.0,        // 10%
            duration: 120.0,         // 2 minutes
            target_mi: 0.3,          // Low MI for BBB opening
            bubble_size: (1.5, 0.3), // 1.5 ± 0.3 μm
        }
    }
}

/// Treatment protocol for BBB opening
#[derive(Debug, Clone)]
pub struct TreatmentProtocol {
    pub frequency: f64,        // Hz
    pub target_mi: f64,        // MI
    pub duration: f64,         // seconds
    pub prf: f64,              // Hz
    pub duty_cycle: f64,       // %
    pub microbubble_dose: f64, // μL/kg
    pub safety_checks: Vec<String>,
}

/// Safety validation results
#[derive(Debug)]
pub struct SafetyValidation {
    pub max_mechanical_index: f64,
    pub average_enhancement: f64,
    pub is_safe: bool,
    pub warnings: Vec<String>,
}
