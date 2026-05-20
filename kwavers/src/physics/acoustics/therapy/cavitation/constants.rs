//! Physical constants for cavitation threshold models.
//!
//! Re-exports surface-tension and vapour-pressure values from the SSOT
//! `core::constants::cavitation` rather than duplicating them locally.

use crate::core::constants::cavitation::{SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER};
use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;

/// Reference nucleus radius for threshold calculations (1 µm).
pub(super) const DEFAULT_NUCLEUS_RADIUS: f64 = 1e-6;
/// Polytropic index for air.
pub(super) const AIR_POLYTROPIC_INDEX: f64 = 1.4;
/// Quality factor for a µm air bubble in water (Leighton 1994, §4.4).
pub(super) const BUBBLE_Q_FACTOR: f64 = 2.0;
/// Logistic cavitation-probability steepness around CI = 1.
pub(super) const CAVITATION_PROBABILITY_STEEPNESS: f64 = 5.0;

pub(super) fn blake_threshold(r0: f64) -> f64 {
    (ATMOSPHERIC_PRESSURE + VAPOR_PRESSURE_WATER - 2.0 * SURFACE_TENSION_WATER / r0).abs()
}
