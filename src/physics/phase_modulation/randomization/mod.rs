//! Phase Randomization for Standing Wave Suppression
//!
//! Implements phase randomization techniques to reduce standing waves and
//! improve spatial uniformity of acoustic fields.
//!
//! References:
//! - Tang & Clement (2010): "Standing wave suppression for transcranial ultrasound"
//! - Liu et al. (2018): "Random phase modulation for reduction of peak to average power ratio"
//! - Guo et al. (2020): "Reduced cavitation threshold using phase shift keying"

pub mod constants;
pub mod distribution;
pub mod psk;
pub mod randomizer;
pub mod scheme;
pub mod spatial;
pub mod temporal;

// Re-exports
pub use crate::physics::constants::*;
pub use distribution::PhaseDistribution;
pub use psk::PhaseShiftKeying;
pub use randomizer::PhaseRandomizer;
pub use scheme::RandomizationScheme;
pub use spatial::SpatialRandomization;
pub use temporal::TimeRandomization;
