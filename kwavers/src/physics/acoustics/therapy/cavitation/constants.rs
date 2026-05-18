//! Physical constants for cavitation threshold models.

use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;

/// Water surface tension at 20 °C (N/m).
pub(super) const WATER_SURFACE_TENSION: f64 = 0.0728;
/// Water vapour pressure at 20 °C (Pa).
pub(super) const WATER_VAPOR_PRESSURE: f64 = 2.34e3;
/// Reference nucleus radius for threshold calculations (1 µm).
pub(super) const DEFAULT_NUCLEUS_RADIUS: f64 = 1e-6;
/// Polytropic index for air.
pub(super) const AIR_POLYTROPIC_INDEX: f64 = 1.4;
/// Quality factor for a µm air bubble in water (Leighton 1994, §4.4).
pub(super) const BUBBLE_Q_FACTOR: f64 = 2.0;
/// Logistic cavitation-probability steepness around CI = 1.
pub(super) const CAVITATION_PROBABILITY_STEEPNESS: f64 = 5.0;

pub(super) fn blake_threshold(r0: f64) -> f64 {
    (ATMOSPHERIC_PRESSURE + WATER_VAPOR_PRESSURE - 2.0 * WATER_SURFACE_TENSION / r0).abs()
}
