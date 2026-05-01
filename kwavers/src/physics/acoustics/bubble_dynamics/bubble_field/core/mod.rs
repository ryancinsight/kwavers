//! Bubble field management — coupled multi-bubble time integration
//!
//! Manages a spatial collection of bubbles and advances them through time using
//! the Keller-Miksis ODE with secondary Bjerknes pressure coupling.
//!
//! ## Secondary Bjerknes Pressure
//!
//! For a spherical bubble i radiating as a monopole, the near-field pressure at
//! neighbouring bubble j is:
//!
//! ```text
//! p_ij = − ρ_L [ R_i² R̈_i + 2 R_i Ṙ_i² ] / d_ij
//! ```
//!
//! The total effective driving pressure is:
//!
//! ```text
//! p_eff_j = p_acoustic_j + Σ_{i≠j} p_ij
//! ```
//!
//! Coupling is skipped when `R_i / d_ij < coupling_threshold`.

mod accessors;
mod constants;
mod coupling;
mod model;
mod stats;
mod update;

#[cfg(test)]
mod tests;

pub use model::BubbleField;
pub use stats::BubbleFieldStats;
