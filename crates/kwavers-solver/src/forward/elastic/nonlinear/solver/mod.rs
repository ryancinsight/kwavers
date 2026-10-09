//! Nonlinear elastic wave solver with harmonic generation
//!
//! Solves the nonlinear elastic wave equation:
//! ∂²u/∂t² = c²∇²u + β c² u/u_ref ∇²u + source terms
//!
//! ## Numerical Methods
//! - **Spatial discretization**: Second-order finite differences
//! - **Time integration**: Second-order Runge-Kutta (Heun's method)
//! - **Shock capturing**: Minmod flux limiter for nonlinear waves
//! - **Stability**: CFL condition with adaptive time stepping
//!
//! ## References
//! - LeVeque, R. J. (2002). "Finite Volume Methods for Hyperbolic Problems", Cambridge.
//! - Chen, S., et al. (2013). "Harmonic motion detection in ultrasound elastography."
//!   IEEE Trans. Medical Imaging, 32(5), 863-874.

mod core;
mod harmonics;
mod propagation;
mod stability;
mod stepping;
#[cfg(test)]
mod tests;

pub use core::NonlinearElasticWaveSolver;
