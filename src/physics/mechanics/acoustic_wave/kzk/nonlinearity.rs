//! Nonlinearity operator for KZK equation
//!
//! Implements the quadratic nonlinearity term.
//! Reference: Hamilton & Morfey (1998) "Model equations"

use super::KZKConfig;
use ndarray::Array3;

/// Nonlinear operator for KZK equation
pub struct NonlinearOperator {
    /// Nonlinearity coefficient
    beta: f64,
    /// Configuration
    config: KZKConfig,
}

impl NonlinearOperator {
    /// Create new nonlinear operator
    pub fn new(config: &KZKConfig) -> Self {
        Self {
            beta: 1.0 + config.beta / 2.0, // β = 1 + B/2A
            config: config.clone(),
        }
    }

    /// Apply nonlinearity for one step
    /// Solves: ∂p/∂z = (β/2ρ₀c₀³) * p * ∂p/∂τ
    pub fn apply(
        &mut self,
        pressure: &mut Array3<f64>,
        pressure_prev: &Array3<f64>,
        step_size: f64,
    ) {
        let dt = self.config.dt;
        let coeff = self.beta * step_size / (2.0 * self.config.rho0 * self.config.c0.powi(3));

        // Compute time derivative and apply nonlinear correction
        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                for t in 1..self.config.nt - 1 {
                    // Central difference for time derivative
                    let dp_dt = (pressure[[i, j, t + 1]] - pressure[[i, j, t - 1]]) / (2.0 * dt);

                    // Nonlinear correction: Δp = coeff * p * ∂p/∂τ
                    let p = pressure[[i, j, t]];
                    let correction = coeff * p * dp_dt;

                    pressure[[i, j, t]] += correction;
                }
            }
        }
    }

    /// Calculate shock formation distance
    /// z_shock = ρ₀c₀³/(βωp₀) for plane wave
    pub fn shock_distance(&self, frequency: f64, amplitude: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        self.config.rho0 * self.config.c0.powi(3) / (self.beta * omega * amplitude)
    }

    /// Calculate Gol'dberg number (nonlinearity strength)
    /// Γ = z/z_shock
    pub fn goldberg_number(&self, z: f64, frequency: f64, amplitude: f64) -> f64 {
        z / self.shock_distance(frequency, amplitude)
    }
}
