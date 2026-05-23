//! Physical constants for sonoluminescence detection.

use crate::core::constants::numerical::MPA_TO_PA;

/// Minimum temperature for sonoluminescence (K)
/// Based on Brenner et al. (2002)
pub const MIN_TEMPERATURE_SL: f64 = 5000.0;

/// Maximum compression ratio for SL detection
/// Based on Yasui (1997)
pub const MAX_COMPRESSION_RATIO: f64 = 10.0;

/// Minimum pressure for SL (Pa) — 1 MPa threshold
/// Based on experimental observations
pub const MIN_PRESSURE_SL: f64 = MPA_TO_PA;

/// Time window for event detection (s)
pub const EVENT_TIME_WINDOW: f64 = 1e-9;

/// Minimum photon count for detection
pub const MIN_PHOTON_COUNT: f64 = 1e3;
