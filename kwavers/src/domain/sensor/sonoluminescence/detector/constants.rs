//! Physical constants for sonoluminescence detection.

/// Minimum temperature for sonoluminescence (K)
/// Based on Brenner et al. (2002)
pub const MIN_TEMPERATURE_SL: f64 = 5000.0;

/// Maximum compression ratio for SL detection
/// Based on Yasui (1997)
pub const MAX_COMPRESSION_RATIO: f64 = 10.0;

/// Minimum acoustic pressure threshold for sonoluminescence onset (Pa).
///
/// Value: 1.0 MPa (= 1.0e6 Pa).
/// Reference: Brenner, Hilgenfeldt & Lohse (2002), Rev. Mod. Phys. 74, 425.
pub const MIN_PRESSURE_SL: f64 = 1.0e6;

/// Time window for event detection (s)
pub const EVENT_TIME_WINDOW: f64 = 1e-9;

/// Minimum photon count for detection
pub const MIN_PHOTON_COUNT: f64 = 1e3;
