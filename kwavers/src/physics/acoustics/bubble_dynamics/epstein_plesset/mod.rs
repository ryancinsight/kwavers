//! Epstein-Plesset Stability Theorem for Bubble Oscillations
//!
//! Analyzes the stability of small-amplitude bubble oscillations around
//! equilibrium through linear perturbation analysis of the Rayleigh-Plesset equation.
//!
//! ## References
//! - Epstein, P. S., & Plesset, M. S. (1953). J. Chem. Phys., 18(11), 1505-1509.
//! - Prosperetti, A. (1984). Appl. Sci. Res., 38(3), 145-164.

pub mod boundary;
pub mod solver;
pub mod types;
pub mod validation;

#[cfg(test)]
mod tests;

pub use solver::EpsteinPlessetStabilitySolver;
pub use types::{
    AmplitudeEvolution, OscillationType, StabilityAnalysis, StabilityBoundary, ValidationResults,
};
