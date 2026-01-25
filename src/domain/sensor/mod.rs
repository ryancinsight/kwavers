// sensor/mod.rs
//
// This bounded context owns all sensor/observation semantics.
//
// NOTE: Legacy generic "SensorConfig/SensorData" types previously lived here but
// collided semantically with domain-specific sensor recording types. They have been
// removed to enforce a single source of truth and prevent accidental misuse.

pub mod beamforming;
pub mod grid_sampling;
pub mod localization;
pub mod passive_acoustic_mapping; // Multi-lateration localization system
pub mod recorder; // Shared sensor recording logic
pub mod sonoluminescence; // Sonoluminescence detector

// Ultrafast ultrasound imaging (plane wave compounding, high frame rate)
// Based on: Nouhoum et al. (2021), Tanter & Fink (2014), Montaldo et al. (2009)
pub mod ultrafast;

// Canonical high-level probe set (supports both acoustics + optics).
pub use grid_sampling::{GridPoint, GridSensorSet};

pub use localization::{
    array::Sensor as LocalizationSensor, ArrayGeometry as LocalizationArrayGeometry,
    LocalizationResult, SensorArray,
};
// NOTE: Do NOT re-export `localization::array::Sensor` as `Sensor` here.
// It collides semantically with other sensor concepts and previously masked dead APIs.

// Re-export PAM components without colliding names; configs remain module-scoped.
pub use passive_acoustic_mapping::{ArrayElement, ArrayGeometry, DirectivityPattern};
// Expose unified beamforming config at the sensor module root.
pub use beamforming::{BeamformingConfig, BeamformingCoreConfig};
