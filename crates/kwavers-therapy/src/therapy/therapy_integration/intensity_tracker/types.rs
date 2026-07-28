//! Types for acoustic intensity tracking.

use aequitas::systems::si::quantities::{
    Intensity, TemperatureDifference, ThermodynamicTemperature, Time,
};
use kwavers_core::constants::thermodynamic::{BODY_TEMPERATURE_C, KELVIN_OFFSET_C};
use kwavers_physics::thermal::CumulativeEquivalentMinutes;

/// Acoustic intensity measurement at a point in time
#[derive(Debug, Clone, Copy)]
pub struct InstantaneousIntensity {
    /// Spatial peak pulse average (W/m²)
    pub isppa: Intensity<f64>,
    /// Spatial peak (instantaneous maximum)
    pub spatial_peak: Intensity<f64>,
    /// Spatial average within focal region
    pub spatial_average: Intensity<f64>,
    /// Measurement time (seconds)
    pub timestamp: Time<f64>,
}

/// Temporal-averaged intensity metrics
#[derive(Debug, Clone, Copy)]
pub struct TemporalIntensityMetrics {
    /// Spatial peak temporal average (FDA metric, W/m²)
    pub spta: Intensity<f64>,
    /// Temporal average spatial average
    pub tas: Intensity<f64>,
    /// Peak measured SPTA within monitoring window
    pub peak_spta: Intensity<f64>,
    /// Minimum SPTA (usually near zero)
    pub min_spta: Intensity<f64>,
    /// Number of measurements averaged
    pub sample_count: usize,
}

impl Default for TemporalIntensityMetrics {
    fn default() -> Self {
        Self {
            spta: Intensity::from_base(0.0),
            tas: Intensity::from_base(0.0),
            peak_spta: Intensity::from_base(0.0),
            min_spta: Intensity::from_base(f64::MAX),
            sample_count: 0,
        }
    }
}

/// Thermal dose tracking (CEM43 model)
#[derive(Debug, Clone, Copy)]
pub struct IntensityTrackerDose {
    /// Cumulative equivalent minutes at 43°C
    pub cem43: CumulativeEquivalentMinutes,
    /// Current temperature (°C)
    pub current_temperature: ThermodynamicTemperature<f64>,
    /// Maximum temperature recorded (°C)
    pub max_temperature: ThermodynamicTemperature<f64>,
    /// Temperature rise above baseline (°C)
    pub temperature_rise: TemperatureDifference<f64>,
}

impl Default for IntensityTrackerDose {
    fn default() -> Self {
        Self {
            cem43: CumulativeEquivalentMinutes::zero(),
            current_temperature: ThermodynamicTemperature::from_base(
                BODY_TEMPERATURE_C + KELVIN_OFFSET_C,
            ),
            max_temperature: ThermodynamicTemperature::from_base(
                BODY_TEMPERATURE_C + KELVIN_OFFSET_C,
            ),
            temperature_rise: TemperatureDifference::from_base(0.0),
        }
    }
}
