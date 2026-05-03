//! Unified SIRT/ART/OSEM reconstruction interface.

pub mod config;
pub mod reconstructor;
#[cfg(test)]
mod tests;

pub use config::{SirtAlgorithm, SirtConfig, SirtResult};
pub use reconstructor::SirtReconstructor;
