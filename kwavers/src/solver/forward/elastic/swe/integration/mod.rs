//! Time integration schemes for elastic wave propagation.
//!
//! Implements velocity-Verlet (leapfrog) time integration for the elastic wave equation.
//!
//! ## Velocity-Verlet Scheme
//!
//! Second-order accurate symplectic integrator:
//! ```text
//! v(t+Δt/2) = v(t) + (Δt/2) * a(t)
//! u(t+Δt)   = u(t) + Δt * v(t+Δt/2)
//! v(t+Δt)   = v(t+Δt/2) + (Δt/2) * a(t+Δt)
//! ```
//!
//! ## References
//!
//! - Verlet, L. (1967). "Computer experiments on classical fluids." *Phys. Rev.*, 159(1), 98.

pub mod integrator;

#[cfg(test)]
mod tests;

pub use integrator::TimeIntegrator;
