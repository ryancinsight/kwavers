//! Broadband time-domain **memory-variable** viscoacoustic wave solver.
//!
//! This is the time-domain realization of the generalized-Maxwell viscoacoustic
//! medium ([`kwavers_medium::viscoelastic::GeneralizedMaxwellModel`] supplies the
//! frequency-domain complex modulus). Unlike the fractional-Laplacian path
//! (which realizes the absorption at a single drive frequency), this solver
//! reproduces the **full broadband** absorption and dispersion of the relaxation
//! spectrum exactly, by carrying one memory variable per relaxation arm.
//!
//! ## Model
//! For `L` Maxwell arms (strength `ΔMₗ`, relaxation time `τₗ`) over an
//! equilibrium modulus `M_∞`, the closed first-order system is
//!
//! ```text
//!   ∂v/∂t  = -(1/ρ) ∂p/∂x
//!   ∂p/∂t  = -M_U (∂v/∂x) - Σₗ σₗ/τₗ              M_U = M_∞ + Σₗ ΔMₗ
//!   ∂σₗ/∂t = -σₗ/τₗ - ΔMₗ (∂v/∂x)
//! ```
//!
//! Each arm `σₗ` is a standard Maxwell element driven by the dilatation rate
//! `∂v/∂x`; eliminating the `σₗ` in the frequency domain recovers exactly
//! `M(ω) = M_∞ + Σₗ ΔMₗ · iωτₗ/(1+iωτₗ)`, so the plane-wave absorption and phase
//! velocity match `GeneralizedMaxwellModel` across the whole band (verified in
//! the tests by the complex dispersion `ρω² = M(ω)k²`).
//!
//! ## Numerics
//! - Spatial derivatives are **pseudospectral** (FFT, exact for band-limited
//!   fields), reusing the shared 1-D FFT plan cache.
//! - The wave part uses a velocity–pressure **leapfrog** (`v` at half steps).
//! - The stiff relaxation ODE is advanced with an **exact exponential
//!   integrator** (`σₗ ← e^{-Δt/τₗ}σₗ - ΔMₗ D τₗ(1-e^{-Δt/τₗ})`), so `Δt` is
//!   limited only by the CFL of the unrelaxed wave speed `√(M_U/ρ)`, never by
//!   the smallest `τₗ`. The relaxation contribution to the pressure update is
//!   trapezoidal in `σₗ` (second order).
//! - All work buffers are preallocated; a step performs no heap allocation.
//!
//! ## References
//! - Carcione, J.M. (2007). *Wave Fields in Real Media*, 2nd ed., Ch. 3.
//! - Robertsson, J.O.A., Blanch, J.O. & Symes, W.W. (1994). "Viscoelastic
//!   finite-difference modeling." *Geophysics* 59(9), 1444–1456.
//! - Blanch, J.O., Robertsson, J.O.A. & Symes, W.W. (1995). "Modeling of a
//!   constant Q." *Geophysics* 60(1), 176–184.

mod solver;
#[cfg(test)]
mod tests;

pub use solver::ViscoacousticMemorySolver;
