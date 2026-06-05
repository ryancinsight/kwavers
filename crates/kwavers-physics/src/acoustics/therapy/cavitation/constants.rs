//! Physical constants for cavitation threshold models.
//!
//! Re-exports surface-tension and vapour-pressure values from the SSOT
//! `core::constants::cavitation` rather than duplicating them locally.

pub(super) use kwavers_core::constants::acoustic_parameters::AIR_POLYTROPIC_INDEX;
use kwavers_core::constants::cavitation::{SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER};
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

/// Reference nucleus radius for threshold calculations (1 µm).
pub(super) const DEFAULT_NUCLEUS_RADIUS: f64 = 1e-6;
/// Quality factor for a µm air bubble in water (Leighton 1994, §4.4).
pub(super) const BUBBLE_Q_FACTOR: f64 = 2.0;
/// Logistic cavitation-probability steepness around CI = 1.
pub(super) const CAVITATION_PROBABILITY_STEEPNESS: f64 = 5.0;

/// Rigorous Blake acoustic-amplitude threshold for a gas nucleus of radius `r0`
/// in water, delegating to the canonical implementation
/// [`crate::acoustics::mechanics::cavitation::core::thresholds::blake_threshold`]
/// (SSOT). Returns the rarefaction amplitude (Pa) above which the nucleus grows
/// explosively; it decreases with `r0` because smaller nuclei are more strongly
/// surface-tension-stabilised (Blake 1949).
///
/// (Previously a crude static estimate `|P₀ + P_v − 2σ/R₀|` whose `abs()` produced
/// a non-monotonic, physically wrong dependence on `R₀`.)
pub(super) fn blake_threshold(r0: f64) -> f64 {
    crate::acoustics::mechanics::cavitation::core::thresholds::blake_threshold(
        SURFACE_TENSION_WATER,
        r0,
        ATMOSPHERIC_PRESSURE,
        VAPOR_PRESSURE_WATER,
    )
}
