//! Monolithic solver configuration types.
//!
//! The configuration tree separates report data, Newton-Krylov numerical
//! controls, and physical coefficients.  Validation lives with the data it
//! proves, so solver entry points can reject invalid physics before residual
//! assembly reaches divisions or diffusion coefficients.

mod convergence;
mod newton;
mod physics;

pub use convergence::CouplingConvergenceInfo;
pub use newton::NewtonKrylovConfig;
pub use physics::PhysicsCoefficients;

#[cfg(test)]
mod tests;
