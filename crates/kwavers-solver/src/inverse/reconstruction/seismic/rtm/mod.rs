//! Reverse Time Migration (RTM) implementation
//!
//! Based on:
//! - Baysal et al. (1983): "Reverse time migration"
//! - Claerbout (1985): "Imaging the Earth's Interior"
//! - Zhang & Sun (2009): "Practical issues in reverse time migration"

mod inherent;
mod types;

pub use types::ReverseTimeMigration;
