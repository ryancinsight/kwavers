//! Core Elastic Wave Solver: orchestrates time-domain elastic wave propagation.

mod solver;
#[cfg(test)]
mod tests;

pub use solver::ElasticWaveSolver;
