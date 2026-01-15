//! Photoacoustic multi-physics solver
//!
//! This module implements the coupled photoacoustic solver that integrates
//! electromagnetic (optical) and acoustic physics.

use crate::core::error::KwaversResult;
use crate::domain::field::EMFields;
use crate::physics::electromagnetic::equations::{
    EMMaterialDistribution, ElectromagneticWaveEquation, PhotoacousticCoupling,
};
use crate::physics::electromagnetic::photoacoustic::{
    GruneisenParameter, OpticalAbsorption,
};
use ndarray::ArrayD;

/// Photoacoustic solver implementation
pub struct PhotoacousticSolver<T: ElectromagneticWaveEquation> {
    /// Electromagnetic wave solver
    pub em_solver: T,
    /// Grüneisen parameter for thermoelastic coupling
    pub gruneisen: GruneisenParameter,
    /// Optical absorption properties
    pub optical_properties: OpticalAbsorption,
    /// Initial acoustic pressure field
    pub initial_pressure: Option<ArrayD<f64>>,
}

impl<T: ElectromagneticWaveEquation> std::fmt::Debug for PhotoacousticSolver<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotoacousticSolver")
            .field("gruneisen", &self.gruneisen)
            .field("optical_properties", &self.optical_properties)
            .field("has_initial_pressure", &self.initial_pressure.is_some())
            .finish()
    }
}

impl<T: ElectromagneticWaveEquation> PhotoacousticSolver<T> {
    /// Create a new photoacoustic solver
    pub fn new(
        em_solver: T,
        gruneisen: GruneisenParameter,
        optical_properties: OpticalAbsorption,
    ) -> Self {
        Self {
            em_solver,
            gruneisen,
            optical_properties,
            initial_pressure: None,
        }
    }

    /// Compute initial pressure distribution from optical fluence
    pub fn compute_initial_pressure(
        &mut self,
        fluence: &ArrayD<f64>,
    ) -> KwaversResult<ArrayD<f64>> {
        let gamma = self.gruneisen.get_value(310.0, 1e5); // Body temperature, atmospheric pressure
        let mu_a = self.optical_properties.absorption_coefficient;

        // p₀ = Γ μ_a Φ
        let pressure = fluence.mapv(|phi| gamma * mu_a * phi);

        self.initial_pressure = Some(pressure.clone());
        Ok(pressure)
    }

    /// Compute optical fluence using diffusion approximation
    pub fn compute_fluence_diffusion(
        &self,
        _source_position: &[f64],
        evaluation_points: &ArrayD<f64>,
    ) -> KwaversResult<ArrayD<f64>> {
        // Simplified diffusion approximation
        // Φ(r) ∝ exp(-μ_eff r)/r where μ_eff = √(3 μ_a μ_s')

        let mu_a = self.optical_properties.absorption_coefficient;
        let mu_s_prime = self.optical_properties.reduced_scattering;
        let _mu_eff = (3.0 * mu_a * (mu_a + mu_s_prime)).sqrt();

        // This is a placeholder - real implementation would need proper spatial computation
        let fluence = ArrayD::from_elem(evaluation_points.raw_dim(), 1.0);
        Ok(fluence)
    }

    /// Get acoustic energy deposited by photoacoustic effect
    pub fn acoustic_energy_deposited(&self) -> Option<f64> {
        self.initial_pressure.as_ref().map(|pressure| {
            // E = (1/(2ρc²)) ∫ p² dV (approximate acoustic energy)
            let rho = 1000.0; // kg/m³
            let c = 1500.0; // m/s

            0.5 / (rho * c * c) * pressure.iter().map(|&p| p * p).sum::<f64>()
        })
    }
}

impl<T: ElectromagneticWaveEquation> ElectromagneticWaveEquation for PhotoacousticSolver<T> {
    fn em_dimension(&self) -> crate::physics::electromagnetic::equations::EMDimension {
        self.em_solver.em_dimension()
    }

    fn material_properties(&self) -> &EMMaterialDistribution {
        self.em_solver.material_properties()
    }

    fn em_fields(&self) -> &EMFields {
        self.em_solver.em_fields()
    }

    fn step_maxwell(&mut self, dt: f64) -> Result<(), String> {
        self.em_solver.step_maxwell(dt)
    }

    fn apply_em_boundary_conditions(&mut self, fields: &mut EMFields) {
        self.em_solver.apply_em_boundary_conditions(fields)
    }

    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String> {
        self.em_solver.check_em_constraints(fields)
    }
}

impl<T: ElectromagneticWaveEquation> PhotoacousticCoupling for PhotoacousticSolver<T> {
    fn optical_absorption(&self, _position: &[f64]) -> f64 {
        self.optical_properties.absorption_coefficient
    }

    fn gruneisen_parameter(&self, _position: &[f64]) -> f64 {
        self.gruneisen.get_value(310.0, 1e5)
    }

    fn reduced_scattering(&self, _position: &[f64]) -> f64 {
        self.optical_properties.reduced_scattering
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::solver::forward::fdtd::ElectromagneticFdtdSolver;
    use crate::physics::electromagnetic::equations::EMMaterialDistribution;

    #[test]
    fn test_photoacoustic_solver() {
        // Create mock EM solver
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        // Use canonical domain composition pattern
        let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);

        // Use standard FDTD solver from solver module
        let em_solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();

        // Create photoacoustic solver
        let gruneisen = GruneisenParameter::new(0.5);
        let optical_props = OpticalAbsorption::new(10.0, 50.0, 0.9, 800e-9);

        let mut pa_solver = PhotoacousticSolver::new(em_solver, gruneisen, optical_props);

        // Test fluence to pressure conversion
        let fluence = ArrayD::from_elem(ndarray::IxDyn(&[5, 5, 5]), 100.0); // 100 J/m²
        let pressure = pa_solver.compute_initial_pressure(&fluence).unwrap();

        // Pressure should be positive and proportional to fluence
        assert!(pressure.iter().all(|&p| p > 0.0));
        assert!(pressure.iter().any(|&p| p > 10.0)); // Should be significant pressure
    }
}
