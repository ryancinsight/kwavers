//! Velocity field statistics for k-Wave parity recording modes.
//!
//! Provides three sub-modules:
//! - `accumulator`: `VelocityComponentStats` — online max/min/RMS accumulator.
//! - `sampled`:     `SampledVelocityStats` — sampled statistics at sensor positions.
//! - `interpolation`: `interpolate_staggered_to_collocated` — half-cell shift.

mod accumulator;
mod helpers;
mod interpolation;
mod sampled;

#[cfg(test)]
mod tests;

pub use accumulator::{VelocityArray3Access, VelocityComponentStats};
pub use interpolation::interpolate_staggered_to_collocated;
pub use sampled::SampledVelocityStats;
