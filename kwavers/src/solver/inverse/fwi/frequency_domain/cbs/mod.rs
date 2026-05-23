//! Convergent Born-series machinery for frequency-domain FWI.
//!
//! This module owns the CBS algebra and the first real volume-field kernel used
//! by the frequency-domain FWI path. It is intentionally split by theorem:
//! scattering potential, grid/source projection, shifted Green operator, and
//! fixed-point solve.

mod absorbing;
mod green;
pub mod grid;
mod potential;
mod projection;
mod solve;
mod spectral;

#[cfg(test)]
mod tests;

pub use absorbing::AbsorbingBoundary;
pub(super) use green::apply_shifted_green_adjoint_operator;
pub use green::GreenOperatorKind;
pub use grid::{bli_weights, BliConfig, GridSpec, GridWeight};
pub use potential::{
    convergence_epsilon, pointwise_preconditioner, real_scattering_potential, shifted_potential,
};
pub use projection::{
    receiver_adjoint_from_bli, sample_array_with_bli, sample_field_with_bli,
    source_density_for_operator, source_density_from_bli,
};
pub use solve::{
    solve_adjoint_volume_field, solve_adjoint_volume_field_with_operator, solve_volume_field,
    solve_volume_field_with_operator, CbsConfig, CbsSolution,
};
