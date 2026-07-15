use super::super::{materials::EMMaterialDistribution, types::EMDimension};
use kwavers_core::constants::fundamental::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_field::EMFields;
use leto::{ArrayD, VecStorage};

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
    fn wave_impedance(&self) -> ArrayD<f64, VecStorage<f64>> {
        let props = self.material_properties();
        let shape = props.permittivity.shape().to_vec();
        // η = √(μ/ε) where μ = μ_r * μ₀, ε = ε_r * ε₀ (μ₀ and ε₀ from SSOT).
        let vals: Vec<f64> = props
            .permeability
            .iter()
            .zip(props.permittivity.iter())
            .map(|(&mu_r, &eps_r)| {
                let mu = mu_r * VACUUM_PERMEABILITY;
                let eps = eps_r * VACUUM_PERMITTIVITY;
                (mu / eps).sqrt()
            })
            .collect();
        ArrayD::<f64, VecStorage<f64>>::from_shape_vec(&shape, vals)
            .expect("wave_impedance: shape matches element count")
    }

    /// Compute skin depth δ = √(2/(ωμσ)) (m)
    fn skin_depth(&self, frequency: f64) -> ArrayD<f64, VecStorage<f64>> {
        let props = self.material_properties();
        let omega = TWO_PI * frequency;
        let shape = props.permeability.shape().to_vec();

        let vals: Vec<f64> = props
            .permeability
            .iter()
            .zip(props.conductivity.iter())
            .map(|(&mu_r, &sigma)| {
                let mu = mu_r * VACUUM_PERMEABILITY;
                if sigma > 0.0 {
                    (2.0 / (omega * mu * sigma)).sqrt()
                } else {
                    f64::INFINITY // No skin effect in insulators
                }
            })
            .collect();
        ArrayD::<f64, VecStorage<f64>>::from_shape_vec(&shape, vals)
            .expect("skin_depth: shape matches element count")
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
