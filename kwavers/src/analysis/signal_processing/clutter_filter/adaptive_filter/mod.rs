//! Adaptive Clutter Filter — eigenfilter decomposition and Wiener filtering
//! for optimal SNR in Doppler ultrasound.
//!
//! ## References
//!
//! - Yu & Lovstakken (2010) — eigen-based clutter filter design
//! - Bjaerum et al. (2002) — clutter filter design for ultrasound color flow imaging

mod filter;
#[cfg(test)]
mod tests;
mod types;

pub use filter::AdaptiveFilter;
pub use types::{AdaptiveFilterConfig, CbrEstimationMethod, SubspaceSeparationMethod};
