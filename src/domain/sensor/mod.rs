// sensor/mod.rs
//
// This bounded context owns all sensor/observation semantics.
//
// NOTE: Legacy generic "SensorConfig/SensorData" types previously lived here but
// collided semantically with domain-specific sensor recording types. They have been
// removed to enforce a single source of truth and prevent accidental misuse.

pub mod array; // Sensor array geometry (domain concept - SSOT for sensor positions)
pub mod beamforming;
pub mod grid_sampling;
pub mod localization; // DEPRECATED - use analysis::signal_processing::localization instead
pub mod passive_acoustic_mapping; // Multi-lateration localization system
pub mod recorder; // Shared sensor recording logic
pub mod sonoluminescence; // Sonoluminescence detector

// Ultrafast ultrasound imaging (plane wave compounding, high frame rate)
// Based on: Nouhoum et al. (2021), Tanter & Fink (2014), Montaldo et al. (2009)
pub mod ultrafast;

// Canonical high-level probe set (supports both acoustics + optics).
pub use grid_sampling::{GridPoint, GridSensorSet};

// Sensor array types (domain layer: hardware geometry)
pub use array::{ArrayGeometry, Position, Sensor, SensorArray};

// Re-export PAM components without colliding names; configs remain module-scoped.
pub use passive_acoustic_mapping::{
    ArrayElement, ArrayGeometry as PAMArrayGeometry, DirectivityPattern,
};
// Expose unified beamforming config at the sensor module root.
pub use beamforming::{BeamformingConfig, BeamformingCoreConfig};
