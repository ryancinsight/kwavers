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

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use super::config::NonlinearSWEConfig;
use super::material::HyperelasticModel;
use super::numerics::NumericsOperators;

mod harmonics;
mod propagation;
mod stability;
mod stepping;
#[cfg(test)]
mod tests;

/// Nonlinear elastic wave equation solver
#[derive(Debug)]
pub struct NonlinearElasticWaveSolver {
    pub(super) grid: Grid,
    pub(super) _material: HyperelasticModel,
    pub(super) config: NonlinearSWEConfig,
    pub(super) attenuation_np_per_m: f64,
    pub(super) numerics: NumericsOperators,
}

impl NonlinearElasticWaveSolver {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        material: HyperelasticModel,
        config: NonlinearSWEConfig,
    ) -> KwaversResult<Self> {
        let attenuation_np_per_m = medium
            .optical_absorption_coefficient(0.0, 0.0, 0.0, grid)
            .max(0.0);

        let numerics = NumericsOperators::new(grid.clone());

        Ok(Self {
            grid: grid.clone(),
            _material: material,
            config,
            attenuation_np_per_m,
            numerics,
        })
    }
}
