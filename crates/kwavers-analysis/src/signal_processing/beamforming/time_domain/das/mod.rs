//! Time-domain Delay-and-Sum (DAS) beamforming with explicit delay reference policy.
//!
//! See [`delay_and_sum`] for full documentation and mathematical foundation.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::{delay_and_sum, DEFAULT_DELAY_REFERENCE};
