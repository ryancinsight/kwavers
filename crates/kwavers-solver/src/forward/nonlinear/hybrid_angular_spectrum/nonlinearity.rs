//! Nonlinear operator using Burgers equation
//!
//! Reference: Hamilton & Blackstock (1998) "Nonlinear Acoustics"

use super::HASConfig;
use kwavers_core::error::KwaversResult;
use ndarray::Array3;

/// Nonlinear operator for Burgers equation
#[derive(Debug)]
pub struct HybridAsNonlinearOperator {
    beta: f64,
    c0: f64,
    rho0: f64,
}

impl HybridAsNonlinearOperator {
    /// Create new nonlinear operator
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(config: &HASConfig) -> Self {
        // Convert B/A to β = 1 + B/(2A) (Hamilton & Blackstock 1998 §2.3.2 eq. 2.3.10)
        Self {
            beta: 1.0 + config.nonlinearity / 2.0,
            c0: config.sound_speed,
            rho0: config.density,
        }
    }

    /// Apply nonlinear step
    ///
    /// Solves the lossless Burgers equation in the spatial domain:
    ///
    /// ```text
    /// ∂p/∂z = (β / (ρ₀c₀³)) · p · ∂p/∂τ
    /// ```
    ///
    /// For a forward-propagating plane wave ∂p/∂τ ≈ ∂p/∂t ≈ −c₀ · ∂p/∂z,
    /// giving the spatial update rule:
    ///
    /// ```text
    /// Δp = −(β · Δz / (ρ₀c₀²)) · p · ∂p/∂z
    /// ```
    ///
    /// Reference: Hamilton & Blackstock (1998) §4.2.1, eq. (4.2.3).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply(&self, pressure: &Array3<f64>, dz: f64) -> KwaversResult<Array3<f64>> {
        let mut result = pressure.clone();

        // Coefficient for nonlinear term: −β·Δz / (ρ₀c₀³)
        // (the c₀ factor below converts ∂p/∂z → ∂p/∂τ via the plane-wave
        // approximation ∂p/∂τ ≈ −c₀·∂p/∂z, giving net factor −β·Δz/(ρ₀c₀²))
        let coeff = -self.beta * dz / (self.rho0 * self.c0 * self.c0 * self.c0);

        // Apply nonlinear correction using characteristic method
        for ((i, j, k), val) in pressure.indexed_iter() {
            // Nonlinear steepening: Δp ≈ −β/(ρc³) × p × ∂p/∂τ × Δz
            // ∂p/∂τ ≈ −c₀ × ∂p/∂z for a forward-propagating plane wave
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
