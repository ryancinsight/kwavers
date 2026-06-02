//! TDOA (Time-Difference-of-Arrival) Localization
//!
//! Implements source localization from time-delay estimates between sensor pairs.
//!
//! References:
//! - Knapp, C. H., & Carter, G. C. (1976). "The generalized correlation method for estimation of time delay"
//! - Cafforio, C., & Rocca, F. (1976). "Direction determination in seismic signal processing"

mod functions;
mod processor;
#[cfg(test)]
mod tests;
pub mod types;

pub use processor::TDOAProcessor;
pub use types::{TDOAConfig, TimeDelayMethod};
