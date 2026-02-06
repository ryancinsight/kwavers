//! Modular feedback control system for cavitation
//!
//! This module provides a decomposed, maintainable implementation of
//! feedback control for acoustic cavitation management.

pub mod config;
pub mod controller;
pub mod history;
pub mod safety;
pub mod strategy;

// Re-export main types
pub use config::FeedbackConfig;
pub use controller::FeedbackController;
pub use history::ControlHistory;
pub use safety::SafetyMonitor;
pub use strategy::ControlStrategy;
