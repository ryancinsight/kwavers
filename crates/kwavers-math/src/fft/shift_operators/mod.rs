//! Staggered-grid shift operators for spectral acoustic solvers.

pub mod functions;
#[cfg(test)]
mod tests;

pub use functions::{generate_kappa, generate_shift_1d, generate_source_kappa};
