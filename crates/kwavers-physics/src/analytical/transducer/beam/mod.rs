//! Array geometry, delay laws, beam patterns, steering envelopes, and
//! on-axis pressure for transducer arrays (split from the former beam.rs).
//!
//! Covers: 2-D complex beam pattern (far-field monopole), focusing delay laws,
//! on-axis pressure of circular pistons, and focused spherical bowls.

pub mod beam_pattern;
pub mod delay_laws;
pub mod geometry;
pub mod on_axis;
pub mod steering;

pub use beam_pattern::*;
pub use delay_laws::*;
pub use geometry::*;
pub use on_axis::*;
pub use steering::*;
