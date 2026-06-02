//! Physical constants for CT-derived skull acoustic properties.

/// Hounsfield unit of pure water (calibration reference).
pub const HU_WATER: f64 = 0.0;
/// Hounsfield unit of fully mineralised cortical bone.
pub const HU_CORTICAL: f64 = 1000.0;
/// Water attenuation at clinical frequencies [Np/m/MHz].
pub const ALPHA_WATER: f64 = 0.002;
