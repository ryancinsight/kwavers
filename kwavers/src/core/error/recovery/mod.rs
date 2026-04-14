//! Automated recovery strategies for known error classes.

mod action;
mod attempt;
mod manager;
mod strategies;
mod strategy;

#[cfg(test)]
mod tests;

pub use action::RecoveryAction;
pub use attempt::RecoveryAttempt;
pub use manager::{RecoveryBuilder, RecoveryManager, RecoveryStatistics};
pub use strategies::{CflViolationRecovery, ConvergenceFailureRecovery, GpuOomRecovery};
pub use strategy::{RecoveryResult, RecoveryStrategy};
