//! Configuration types for neural beamforming.

mod adaptation;
mod feature;
mod geometry;
mod mode;
mod neural;
mod physics;
#[cfg(test)]
mod tests;

pub use adaptation::AdaptationParameters;
pub use feature::FeatureConfig;
pub use geometry::SensorGeometry;
pub use mode::NeuralBeamformingMode;
pub use neural::NeuralBeamformingConfig;
pub use physics::PhysicsParameters;
