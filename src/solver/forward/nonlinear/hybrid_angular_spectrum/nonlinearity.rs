//! Nonlinear operator using Burgers equation
//!
//! Reference: Hamilton & Blackstock (1998) "Nonlinear Acoustics"

use super::HASConfig;
use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

/// Nonlinear operator for Burgers equation
#[derive(Debug)]
pub struct NonlinearOperator {
    beta: f64,
    c0: f64,
    rho0: f64,
}

impl NonlinearOperator {
    /// Create new nonlinear operator
    pub fn new(config: &HASConfig) -> Self {
        Self {
            beta: config.nonlinearity,
            c0: config.sound_speed,
            rho0: config.density,
        }
    }

    /// Apply nonlinear step
    ///
    /// Solves ∂p/∂z = -β/(2ρ₀c₀³) × p × ∂p/∂t in time domain
    pub fn apply(&self, pressure: &Array3<f64>, dz: f64) -> KwaversResult<Array3<f64>> {
        let mut result = pressure.clone();

        // Coefficient for nonlinear term
        let coeff = -self.beta * dz / (2.0 * self.rho0 * self.c0 * self.c0 * self.c0);

        // Apply nonlinear correction using characteristic method
        for ((i, j, k), val) in pressure.indexed_iter() {
            // Nonlinear steepening: Δp ≈ -β/(2ρc³) × p × ∂p/∂t × Δz
            // Approximate ∂p/∂t ≈ c₀ × ∂p/∂z (plane wave assumption)
            let dp_dz = if k < pressure.shape()[2] - 1 {
                (pressure[[i, j, k + 1]] - val) / dz
            } else if k > 0 {
                (val - pressure[[i, j, k - 1]]) / dz
            } else {
                0.0
            };

            let nonlinear_term = coeff * val * self.c0 * dp_dz;
            result[[i, j, k]] = val + nonlinear_term;
        }

        Ok(result)
    }
}
