//! PID Controller for Cavitation Feedback Control
//!
//! Implements a discrete-time PID controller with anti-windup and derivative filtering.
//!
//! # References
//! - Åström & Hägglund (2006): "Advanced PID Control"
//! - Franklin et al. (2015): "Feedback Control of Dynamic Systems"

pub mod continuous;
pub mod core;
pub mod discrete;

pub use continuous::PIDController;
pub use core::{ControllerOutput, ErrorIntegral, PIDConfig, PIDGains};
pub use discrete::TustinPIDController;

#[cfg(test)]
mod tests;
