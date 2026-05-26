use super::super::{materials::EMMaterialDistribution, types::EMDimension};
use crate::core::constants::fundamental::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};
use crate::domain::field::EMFields;
use crate::core::constants::numerical::{TWO_PI};

/// Core electromagnetic wave equation trait
///
/// Defines the mathematical structure for solving Maxwell's equations.
/// Implementations can use FDTD, FEM, spectral methods, or analytical solutions.
pub trait ElectromagneticWaveEquation: Send + Sync {
    /// Get electromagnetic spatial dimension
    fn em_dimension(&self) -> EMDimension;

    /// Get material properties at current time
    fn material_properties(&self) -> &EMMaterialDistribution;

    /// Get current electromagnetic fields
    fn em_fields(&self) -> &EMFields;

    /// Compute wave impedance η = √(μ/ε) (Ω)
    fn wave_impedance(&self) -> ndarray::ArrayD<f64> {
        let props = self.material_properties();
        // η = √(μ/ε) where μ = μ_r * μ₀, ε = ε_r * ε₀ (μ₀ and ε₀ from SSOT).
        let mu = &props.permeability * VACUUM_PERMEABILITY;
        let eps = &props.permittivity * VACUUM_PERMITTIVITY;

        ndarray::Zip::from(&mu)
            .and(&eps)
            .map_collect(|&m, &e| (m / e).sqrt())
    }

    /// Compute skin depth δ = √(2/(ωμσ)) (m)
    fn skin_depth(&self, frequency: f64) -> ndarray::ArrayD<f64> {
        let props = self.material_properties();
        let omega = TWO_PI * frequency;

        let mu = &props.permeability * VACUUM_PERMEABILITY;
        let sigma = &props.conductivity;

        // δ = √(2/(ωμσ))
        ndarray::Zip::from(&mu).and(sigma).map_collect(|&m, &s| {
            if s > 0.0 {
                (2.0 / (omega * m * s)).sqrt()
            } else {
                f64::INFINITY // No skin effect in insulators
            }
        })
    }

    /// Solve Maxwell's equations for one time step
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn step_maxwell(&mut self, dt: f64) -> Result<(), String>;

    /// Apply electromagnetic boundary conditions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_em_boundary_conditions(&mut self, fields: &mut EMFields);

    /// Check electromagnetic physics constraints
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String>;
}
