//! Constants for hemispherical array configuration
//!
//! These are physical and engineering constants used in clinical and research applications.
//! Some constants may be unused in current implementations but are kept for completeness.

/// Typical radius for clinical hemispherical arrays (m)
pub const HEMISPHERE_RADIUS_DEFAULT: f64 = 0.15; // 150mm radius (Insightec ExAblate)

/// Half-wavelength element spacing for improved steering (m)
pub const HALF_WAVELENGTH_SPACING: f64 = 1.15e-3; // λ/2 at 650 kHz

/// Maximum steering angle from geometric focus (radians)
pub const MAX_STEERING_ANGLE_RAD: f64 = 0.5236; // 30 degrees

/// Minimum element density for sparse arrays (elements per m²)
pub const MIN_ELEMENT_DENSITY: f64 = 5000.0; // 0.5 per cm²

/// Maximum element density for dense packing (elements per m²)
pub const MAX_ELEMENT_DENSITY: f64 = 40000.0; // 4.0 per cm²

/// Grating lobe threshold (linear ratio below main lobe)
pub const GRATING_LOBE_THRESHOLD_RATIO: f64 = 0.0316; // -30 dB

/// Treatment envelope expansion factor with sparse arrays
pub const ENVELOPE_EXPANSION_FACTOR: f64 = 1.5;

/// Power efficiency threshold for element selection
pub const POWER_EFFICIENCY_THRESHOLD: f64 = 0.7;

/// Minimum f-number for hemispherical arrays
pub const MIN_F_NUMBER: f64 = 0.8;

/// Maximum f-number for hemispherical arrays  
pub const MAX_F_NUMBER: f64 = 1.2;

/// Clinical frequency range (Hz)
pub const CLINICAL_FREQ_MIN: f64 = 200e3; // 200 kHz
pub const CLINICAL_FREQ_MAX: f64 = 2e6; // 2 MHz

/// Skull attenuation coefficient (Np/m/MHz)
pub const SKULL_ATTENUATION: f64 = 70.0;
