//! Special signal types
//!
//! This module contains special signal implementations that are used
//! across different parts of the system, including:
//! - Null signals for testing and placeholders
//! - Time-varying signals with pre-computed values
//! - Composite signals that combine multiple signals

pub mod null_signal;
pub mod time_varying;

pub use null_signal::NullSignal;
pub use time_varying::TimeVaryingSignal;
