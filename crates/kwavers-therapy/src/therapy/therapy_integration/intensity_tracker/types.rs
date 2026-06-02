//! Types for acoustic intensity tracking.

use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;

/// Acoustic intensity measurement at a point in time
#[derive(Debug, Clone, Copy)]
pub struct InstantaneousIntensity {
    /// Spatial peak pulse average (W/m²)
    pub isppa: f64,
    /// Spatial peak (instantaneous maximum)
    pub spatial_peak: f64,
    /// Spatial average within focal region
    pub spatial_average: f64,
    /// Measurement time (seconds)
    pub timestamp: f64,
}

/// Temporal-averaged intensity metrics
#[derive(Debug, Clone, Copy)]
pub struct TemporalIntensityMetrics {
    /// Spatial peak temporal average (FDA metric, W/m²)
    pub spta: f64,
    /// Temporal average spatial average
    pub tas: f64,
    /// Peak measured SPTA within monitoring window
    pub peak_spta: f64,
    /// Minimum SPTA (usually near zero)
    pub min_spta: f64,
    /// Number of measurements averaged
    pub sample_count: usize,
}

impl Default for TemporalIntensityMetrics {
    fn default() -> Self {
        Self {
            spta: 0.0,
            tas: 0.0,
            peak_spta: 0.0,
            min_spta: f64::MAX,
            sample_count: 0,
        }
    }
}

/// Thermal dose tracking (CEM43 model)
#[derive(Debug, Clone, Copy)]
pub struct IntensityTrackerDose {
    /// Cumulative equivalent minutes at 43°C
    pub cem43: f64,
    /// Current temperature (°C)
    pub current_temperature: f64,
    /// Maximum temperature recorded (°C)
    pub max_temperature: f64,
    /// Temperature rise above baseline (°C)
    pub temperature_rise: f64,
}

impl Default for IntensityTrackerDose {
    fn default() -> Self {
        Self {
            cem43: 0.0,
            current_temperature: BODY_TEMPERATURE_C, // Normal body temperature
            max_temperature: BODY_TEMPERATURE_C,
            temperature_rise: 0.0,
        }
    }
}
