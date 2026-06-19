//! Time-domain Delay-and-Sum (DAS) beamforming with explicit delay reference policy.
//!
//! See [`delay_and_sum`] for full documentation and mathematical foundation.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::{align_channels, delay_and_sum, sum_aligned, DEFAULT_DELAY_REFERENCE};
