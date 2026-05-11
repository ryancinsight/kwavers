//! Conservation Law Checkers
//!
//! Core module for verifying conservation of physical quantities during simulation.
//!
//! Partitioned by responsibility:
//! - `types`   — `ConservationLaw`, `ConservedQuantity`, `ConservationResult`.
//! - `checker` — `ConservationChecker` struct and full impl.

mod checker;
mod types;
#[cfg(test)]
mod tests;

pub use checker::ConservationChecker;
pub use types::{ConservationLaw, ConservationResult, ConservedQuantity};
