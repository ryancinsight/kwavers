//! Sixth-order central-difference facade.
//!
//! The implementation lives in `core.rs`; value-semantic tests live in
//! `tests.rs`. This facade is the single import surface for
//! `CentralDifference6`.

pub mod core;

pub use core::CentralDifference6;

#[cfg(test)]
mod tests;
