//! Constants for hemispherical array configuration
//!
//! These are physical and engineering constants used in clinical and research applications.
//! Some constants may be unused in current implementations but are kept for completeness.

#![allow(dead_code)] // Clinical configuration constants for library users

use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER;

/// Typical radius for clinical hemispherical arrays (m)
pub const HEMISPHERE_RADIUS_DEFAULT: Length<f64> = Length::from_base(0.15);

/// Half-wavelength element spacing for improved steering (m)
pub const HALF_WAVELENGTH_SPACING: Length<f64> = Length::from_base(1.15e-3);

/// Maximum steering angle from geometric focus (radians)
pub const MAX_STEERING_ANGLE: Angle<f64> = Angle::from_base(std::f64::consts::PI / 6.0);

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
pub const CLINICAL_FREQ_MIN: Frequency<f64> = Frequency::from_base(200e3);
pub const CLINICAL_FREQ_MAX: Frequency<f64> = Frequency::from_base(2.0e6);

/// Nominal sound speed used by the hemispherical-array source model (m/s).
pub const SOUND_SPEED_WATER_NOMINAL: Velocity<f64> = Velocity::from_base(SOUND_SPEED_WATER);

/// Skull attenuation coefficient [Np/(m·MHz)] — Marsac et al. 2017 porosity model upper bound.
///
/// Value: 70 Np/(m·MHz). Canonical SSOT: `acoustic_parameters::SKULL_ATTENUATION_MARSAC_MAX_NP_PER_M_MHZ`.
/// Use that constant directly for new code; this re-declaration is retained here for the
/// hemispherical-array configuration namespace.
pub const SKULL_ATTENUATION: f64 =
    kwavers_core::constants::acoustic_parameters::SKULL_ATTENUATION_MARSAC_MAX_NP_PER_M_MHZ;
