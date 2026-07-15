//! Literature validation types and reference constants.

use std::collections::HashMap;
/// Reference values from Treeby et al. (2010) k-Wave paper
pub mod treeby_2010 {
    use kwavers_core::constants::numerical::TWO_PI;

    pub const SOUND_SPEED: f64 = kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    pub const DENSITY: f64 = kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    pub const FREQUENCY: f64 = kwavers_core::constants::numerical::MHZ_TO_HZ;
    pub const ABSORPTION_COEF: f64 = 0.0;
    pub const GRID_SIZE: (usize, usize, usize) = (128, 128, 128);
    pub const DX: f64 = 1.0e-4;

    pub fn analytical_pressure(t: f64, amplitude: f64) -> f64 {
        let omega = TWO_PI * FREQUENCY;
        let k = omega / SOUND_SPEED;
        let x = 64.0 * DX;
        amplitude * (omega * t - k * x).sin()
    }

    pub const MAX_PHASE_VELOCITY_ERROR: f64 = 0.001;
    pub const MAX_AMPLITUDE_ERROR_DB: f64 = 0.5;
}

/// Reference values from Pinton et al. (2009) elastic wave validation
pub mod pinton_2009 {
    pub const SHEAR_SPEED: f64 = 3.0;
    pub const COMPRESSIONAL_SPEED: f64 = kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
    pub const DENSITY: f64 = kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    pub const FDTD_CONVERGENCE_ORDER: f64 = 2.0;
    pub const FDTD_CFL_STABILITY: f64 = 0.3;
    pub const MAX_ENERGY_ERROR: f64 = 0.01;
}

/// Validation case metadata
#[derive(Debug, Clone)]
pub struct LiteratureValidationCase {
    pub name: &'static str,
    pub paper_citation: &'static str,
    pub scenario: &'static str,
    pub tolerance: f64,
    pub expected_result: ValidationMetric,
}

/// Metric types for validation
#[derive(Debug, Clone)]
pub enum ValidationMetric {
    Scalar(f64),
    Waveform(Vec<f64>),
    Field(Vec<f64>),
    ConvergenceRate(f64),
    EnergyRatio(f64),
}

/// Validation result with detailed error analysis
#[derive(Debug, Clone)]
pub struct LiteratureValidationResult {
    pub case_name: String,
    pub passed: bool,
    pub relative_error: f64,
    pub absolute_error: f64,
    pub error_metrics: HashMap<String, f64>,
    pub diff_field: Option<Vec<f64>>,
    pub notes: String,
}

impl LiteratureValidationResult {
    pub fn new(name: &str) -> Self {
        Self {
            case_name: name.to_string(),
            passed: false,
            relative_error: f64::NAN,
            absolute_error: f64::NAN,
            error_metrics: HashMap::new(),
            diff_field: None,
            notes: String::new(),
        }
    }

    pub fn with_error(&mut self, relative: f64, absolute: f64) -> &mut Self {
        self.relative_error = relative;
        self.absolute_error = absolute;
        self.passed = relative < 0.05;
        self
    }

    pub fn with_metric(&mut self, name: &str, value: f64) -> &mut Self {
        self.error_metrics.insert(name.to_string(), value);
        self
    }
}
