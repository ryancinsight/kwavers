// sensor/adaptive_beamforming/mod.rs - Unified adaptive beamforming

pub mod algorithms;
pub mod array_geometry;
pub mod beamformer;
pub mod steering;
pub mod weights;

// Re-export main types - single implementation
pub use algorithms::{BeamformingAlgorithm, DelayAndSum, MinimumVariance};
pub use array_geometry::{ArrayGeometry, ElementPosition};
pub use beamformer::AdaptiveBeamformer;
pub use steering::{SteeringMatrix, SteeringVector};
pub use weights::{WeightCalculator, WeightingScheme};
