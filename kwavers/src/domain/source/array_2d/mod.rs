//! 2D Transducer Array — Phased Array Beamforming.
//!
//! Native 2D transducer array implementation compatible with k-wave-python's
//! `kWaveTransducerSimple` and `NotATransducer` conventions.
//!
//! ## References
//!
//! - Treeby & Cox (2010) J. Biomed. Opt. 15(2), 021314.
//! - van Veen & Buckley (1988) IEEE Signal Process. Mag. 5(2), 4-24.
//! - Harris (1978) Proc. IEEE 66(1), 51-83.

pub mod builder;

mod array;
mod source_impl;
#[cfg(test)]
mod tests;
mod types;

pub use array::TransducerArray2D;
pub use builder::TransducerArray2DBuilder;
pub use types::{ApodizationType, ArrayElement, TransducerArray2DConfig};
