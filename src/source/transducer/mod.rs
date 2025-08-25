//! Transducer Module
//!
//! Comprehensive transducer design and modeling including element geometry,
//! materials, frequency response, and acoustic characteristics.
//!
//! References:
//! - Szabo (2014): "Diagnostic Ultrasound Imaging: Inside Out"
//! - Shung (2015): "Diagnostic Ultrasound: Imaging and Blood Flow Measurements"
//! - Cobbold (2007): "Foundations of Biomedical Ultrasound"
//! - Kino (1987): "Acoustic Waves: Devices, Imaging, and Analog Signal Processing"
//! - Hunt et al. (1983): "Ultrasound transducers for pulse-echo medical imaging"

pub mod geometry;
pub mod materials;
pub mod frequency;
pub mod directivity;
pub mod coupling;
pub mod sensitivity;
pub mod design;

// Re-export main types for convenience
pub use geometry::ElementGeometry;
pub use materials::{PiezoMaterial, PiezoType, BackingLayer, BackingMaterial, MatchingLayer, AcousticLens, LensMaterial};
pub use frequency::FrequencyResponse;
pub use directivity::DirectivityPattern;
pub use coupling::ElementCoupling;
pub use sensitivity::TransducerSensitivity;
pub use design::TransducerDesign;

// Design constants based on literature
/// Typical piezoelectric coupling coefficient (PZT-5H)
pub const PIEZO_COUPLING_K33: f64 = 0.75;

/// Typical mechanical quality factor
pub const MECHANICAL_Q: f64 = 80.0;

/// Typical electrical quality factor  
pub const ELECTRICAL_Q: f64 = 50.0;

/// Standard acoustic impedance of PZT (MRayl)
pub const PZT_IMPEDANCE: f64 = 30.0;

/// Acoustic impedance of water/tissue (MRayl)
pub const TISSUE_IMPEDANCE: f64 = 1.5;

/// Acoustic impedance of backing material (MRayl)
pub const BACKING_IMPEDANCE: f64 = 5.0;

/// Minimum kerf width as fraction of element width
pub const MIN_KERF_RATIO: f64 = 0.05;

/// Maximum kerf width as fraction of element width
pub const MAX_KERF_RATIO: f64 = 0.3;

/// Typical matching layer thickness (quarter wavelength)
pub const MATCHING_LAYER_FACTOR: f64 = 0.25;

/// Bandwidth threshold (-6 dB) for fractional bandwidth calculation
pub const BANDWIDTH_THRESHOLD_DB: f64 = -6.0;

/// Minimum element aspect ratio (width/thickness)
pub const MIN_ASPECT_RATIO: f64 = 0.5;

/// Maximum element aspect ratio
pub const MAX_ASPECT_RATIO: f64 = 10.0;

/// Typical dielectric constant for PZT
pub const PZT_DIELECTRIC_CONSTANT: f64 = 3400.0;

/// Speed of sound in PZT (m/s)
pub const PZT_SOUND_SPEED: f64 = 4600.0;

/// Typical lens curvature radius factor
pub const LENS_CURVATURE_FACTOR: f64 = 0.7;

/// Maximum steering angle for phased arrays (degrees)
pub const MAX_STEERING_ANGLE: f64 = 45.0;

/// Typical transducer efficiency
pub const TRANSDUCER_EFFICIENCY: f64 = 0.5;

/// Reference pressure for sensitivity calculations (Pa)
pub const REFERENCE_PRESSURE: f64 = 1e-6;

/// Standard test distance for sensitivity (m)
pub const TEST_DISTANCE: f64 = 1.0;