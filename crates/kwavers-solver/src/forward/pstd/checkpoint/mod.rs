//! Binary checkpoint format for PSTD solver mid-simulation state persistence.

pub mod data;
#[cfg(test)]
mod tests;

pub use data::PSTDCheckpoint;
