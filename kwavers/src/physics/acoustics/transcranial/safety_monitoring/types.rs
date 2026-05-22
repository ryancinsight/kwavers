//! Core data structures for safety monitoring

use crate::core::constants::medical::MI_LIMIT_SOFT_TISSUE;
use ndarray::Array3;

/// Thermal dose accumulation (CEM43)
#[derive(Debug, Clone)]
pub struct TranscranialSafetyDose {
    /// Current thermal dose (CEM43)
    pub current_dose: Array3<f64>,
    /// Dose rate (CEM43/s)
    pub dose_rate: Array3<f64>,
    /// Time to reach target dose (s)
    pub time_to_target: Array3<f64>,
    /// Maximum safe exposure time (s)
    pub max_safe_time: Array3<f64>,
}

/// Mechanical index monitoring
#[derive(Debug, Clone)]
pub struct MechanicalIndex {
    /// Current MI value
    pub current_mi: f64,
    /// Peak pressure (MPa)
    pub peak_pressure: f64,
    /// MI limit for tissue type
    pub limit: f64,
    /// Safety margin (`limit / current_mi`; 1.0 = at limit, >1.0 = safe)
    pub safety_margin: f64,
}

/// Safety thresholds for treatment
#[derive(Debug, Clone)]
pub struct SafetyThresholds {
    pub max_temperature: f64,      // °C
    pub max_thermal_dose: f64,     // CEM43
    pub max_mechanical_index: f64, // MI
    pub max_power_density: f64,    // W/cm²
}

impl SafetyThresholds {
    #[must_use]
    pub fn new(
        max_temperature: f64,
        max_thermal_dose: f64,
        max_mechanical_index: f64,
        max_power_density: f64,
    ) -> Self {
        Self {
            max_temperature,
            max_thermal_dose,
            max_mechanical_index,
            max_power_density,
        }
    }
}

impl Default for SafetyThresholds {
    fn default() -> Self {
        Self {
            max_temperature: 43.0,     // Brain tissue limit
            max_thermal_dose: 240.0,   // CEM43 for brain
            max_mechanical_index: MI_LIMIT_SOFT_TISSUE, // FDA limit
            max_power_density: 100.0,  // W/cm²
        }
    }
}

/// Safety level classification
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub enum TranscranialSafetyLevel {
    Safe = 0,
    Monitor = 1,
    Warning = 2,
    Critical = 3,
}

impl TranscranialSafetyLevel {
    #[must_use]
    pub fn from_value(current: f64, limit: f64) -> Self {
        let ratio = current / limit;
        if ratio >= 1.0 {
            Self::Critical
        } else if ratio >= 0.9 {
            Self::Warning
        } else if ratio >= 0.8 {
            Self::Monitor
        } else {
            Self::Safe
        }
    }
}

/// Overall safety status
#[derive(Debug, Clone)]
pub struct TranscranialSafetyStatus {
    pub temperature_status: TranscranialSafetyLevel,
    pub thermal_dose_status: TranscranialSafetyLevel,
    pub mechanical_index_status: TranscranialSafetyLevel,
    pub overall_safety: TranscranialSafetyLevel,
}

/// Treatment progress information
#[derive(Debug, Clone)]
pub struct TreatmentProgress {
    pub dose_progress: f64,            // 0.0 to 1.0
    pub estimated_time_remaining: f64, // seconds
    pub current_max_dose: f64,         // CEM43
    pub target_dose: f64,              // CEM43
}

/// Comprehensive safety report
#[derive(Debug)]
pub struct SafetyReport {
    pub status: TranscranialSafetyStatus,
    pub progress: TreatmentProgress,
    pub thermal_dose: TranscranialSafetyDose,
    pub mechanical_index: MechanicalIndex,
    pub recommendations: Vec<String>,
}
