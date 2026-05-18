//! Delay-and-Sum Passive Acoustic Mapping (DAS-PAM).
//!
//! # Algorithm (Gyöngy & Coussios 2010, IEEE UFFC 57(1))
//!
//! ```text
//! τᵢ = ||r_s − rᵢ|| / c
//! I(r_s) = ∫ |Σᵢ wᵢ · sᵢ(t + τᵢ)|² dt
//! ```

mod processor;
#[cfg(test)]
mod tests;
mod types;

pub use processor::DelayAndSumPAM;
pub use types::{ApodizationType, DelayAndSumConfig, PamCavitationEvent};
