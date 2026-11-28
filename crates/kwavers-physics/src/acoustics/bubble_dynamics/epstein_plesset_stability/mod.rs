//! Epstein–Plesset oscillation-stability analysis for bubble oscillations
//!
//! Analyzes the stability of small-amplitude bubble oscillations around
//! equilibrium through linear perturbation analysis of the Rayleigh-Plesset equation.
//!
//! This module is the oscillation-*stability* side of Epstein &
//! Plesset (1950). It is **not** the Epstein–Plesset dissolution law
//! (the gas-diffusion `dR/dt` equation), which lives in
//! [`super::dissolution`]. The module was renamed from
//! `epstein_plesset` to `epstein_plesset_stability` so the two
//! cannot be confused at import sites (KW-PHYS-009 item 4,
//! 2026-11-27).
//!
//! ## References
//! - Epstein, P. S., & Plesset, M. S. (1950). J. Chem. Phys., 18(11), 1505-1509.
//! - Prosperetti, A. (1984). Appl. Sci. Res., 38(3), 145-164.

pub mod boundary;
pub mod solver;
pub mod types;
pub mod validation;

#[cfg(test)]
mod tests;

pub use solver::EpsteinPlessetStabilitySolver;
pub use types::{
    AmplitudeEvolution, EpsteinPlessetStabilityValidationResults, OscillationType,
    StabilityAnalysis, StabilityBoundary,
};
