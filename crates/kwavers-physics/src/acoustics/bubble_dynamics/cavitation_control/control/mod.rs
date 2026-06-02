//! Modular feedback control components

pub mod adaptive;
pub mod safety_monitor;
pub mod state_estimator;
pub mod types;

pub use adaptive::AdaptiveController;
pub use safety_monitor::CavitationSafetyMonitor;
pub use state_estimator::StateEstimator;
pub use types::{CavitationSafetyLimits, ControlOutput, ControlStrategy, FeedbackConfig};
