//! Feedback control system for cavitation management
//!
//! Modular implementation of negative feedback control for maintaining
//! desired cavitation levels using real-time monitoring.
//!
//! # Architecture
//! - `config`: Configuration and parameters
//! - `strategy`: Control strategy implementations
//! - `controller`: Main feedback controller
//! - `output`: Control output types
//! - `safety`: Safety monitoring and shutdown
//!
//! # References
//! - Hockham et al. (2013): "Real-time control system for sustaining thermally relevant acoustic cavitation"
//! - Arvanitis et al. (2013): "Cavitation-enhanced nonthermal ablation in deep brain targets"

pub mod config;
pub mod controller;
pub mod output;
pub mod safety;
pub mod strategy;

pub use config::{FeedbackConfig, CONTROL_UPDATE_RATE, DEFAULT_TARGET_INTENSITY};
pub use controller::FeedbackController;
pub use output::ControlOutput;
pub use safety::SafetyMonitor;
pub use strategy::{ControlStrategy, StrategyExecutor};