//! PML absorbing boundary conditions for elastic wave propagation.
//!
//! Implements complex-coordinate-stretching absorbing layers that prevent
//! artificial reflections at computational domain boundaries.
//!
//! ## Mathematical Background
//!
//! Coordinate stretching: `∂/∂x → (1 + iσ/ω) ∂/∂x`
//!
//! Time-domain exponential damping: `v(t+Δt) = v(t) * exp(-σ(x) * Δt)`
//!
//! Attenuation profile: `σ(d) = σ_max * (d / L_pml)^n`
//!
//! ## References
//!
//! - Berenger, J. P. (1994). J. Comp. Phys., 114(2), 185–200.
//! - Komatitsch, D., & Martin, R. (2007). Geophysics, 72(5), SM155–SM167.
//! - Collino, F., & Tsogka, C. (2001). Geophysics, 66(1), 294–307.

pub mod config;
pub mod pml;
#[cfg(test)]
mod tests;

pub use config::SwePmlConfig;
pub use pml::PMLBoundary;
