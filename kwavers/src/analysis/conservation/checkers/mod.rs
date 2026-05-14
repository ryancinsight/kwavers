//! Conservation Law Checkers
//!
//! Core module for verifying conservation of physical quantities during simulation.
//!
//! Partitioned by responsibility:
//! - `types`   — `ConservationLaw`, `ConservedQuantity`, `ConservationResult`.
//! - `checker` — `ConservationChecker` struct and full impl.

mod checker;
#[cfg(test)]
mod tests;
mod types;

pub use checker::ConservationChecker;
pub use types::{ConservationLaw, ConservationResult, ConservedQuantity};
