//! Thermal Diffusion Solver Module - Modular Architecture
//!
//! This module implements thermal diffusion solvers with proper separation of concerns
//!
//! This module implements dedicated thermal diffusion solvers including:
//! - Standard heat diffusion equation
//! - Pennes bioheat equation
//! - Thermal dose calculations (CEM43)
//! - Hyperbolic heat transfer (Cattaneo-Vernotte)
//!
//! # Literature References
//!
//! 1. **Pennes, H. H. (1948)**. "Analysis of tissue and arterial blood temperatures
//!    in the resting human forearm." *Journal of Applied Physiology*, 1(2), 93-122.
//!    - Original formulation of bioheat equation
//!
//! 2. **Sapareto, S. A., & Dewey, W. C. (1984)**. "Thermal dose determination in
//!    cancer therapy." *International Journal of Radiation Oncology Biology Physics*,
//!    10(6), 787-800. DOI: 10.1016/0360-3016(84)90379-1
//!    - CEM43 thermal dose formulation
//!
//! 3. **Cattaneo, C. (1958)**. "A form of heat conduction equation which eliminates
//!    the paradox of instantaneous propagation." *Comptes Rendus*, 247, 431-433.
//!    - Hyperbolic heat transfer theory
//!
//! 4. **Liu, J., & Xu, L. X. (1999)**. "Estimation of blood perfusion using phase
//!    shift in temperature response to sinusoidal heating at the skin surface."
//!    *IEEE Transactions on Biomedical Engineering*, 46(9), 1037-1043.
//!    - Modern perfusion estimation methods

// Submodules following SOLID principles
mod bioheat;
mod dose;
mod hyperbolic;
mod solver;

pub use bioheat::{BioheatParameters, PennesBioheat};
pub use dose::{thresholds, ThermalDoseCalculator};
pub use hyperbolic::{CattaneoVernotte, HyperbolicParameters};
pub use solver::ThermalDiffusionSolver;

use crate::{
    error::KwaversResult,
    grid::Grid,
    medium::Medium,
    physics::{
        field_mapping::UnifiedFieldType,
        plugin::{PluginContext, PluginMetadata, PluginState},
    },
};
use ndarray::Array4;

/// Configuration for thermal diffusion solver
#[derive(Debug, Clone)]
pub struct ThermalDiffusionConfig {
    /// Enable Pennes bioheat equation terms
    pub enable_bioheat: bool,
    /// Blood perfusion rate [1/s]
    pub perfusion_rate: f64,
    /// Blood density [kg/m³]
    pub blood_density: f64,
    /// Blood specific heat [J/(kg·K)]
    pub blood_specific_heat: f64,
    /// Arterial blood temperature [K]
    pub arterial_temperature: f64,
    /// Enable hyperbolic heat transfer (Cattaneo-Vernotte)
    pub enable_hyperbolic: bool,
    /// Thermal relaxation time [s]
    pub relaxation_time: f64,
    /// Enable thermal dose tracking
    pub track_thermal_dose: bool,
    /// Reference temperature for dose calculation [°C]
    pub dose_reference_temperature: f64,
    /// Spatial discretization order (2, 4, or 6)
    pub spatial_order: usize,
}

impl Default for ThermalDiffusionConfig {
    fn default() -> Self {
        Self {
            enable_bioheat: true,
            perfusion_rate: 0.5e-3,       // 0.5 mL/g/min typical tissue
            blood_density: 1050.0,        // kg/m³
            blood_specific_heat: 3840.0,  // J/(kg·K)
            arterial_temperature: 310.15, // 37°C in Kelvin
            enable_hyperbolic: false,
            relaxation_time: 20.0, // 20s for tissue
            track_thermal_dose: true,
            dose_reference_temperature: 43.0, // 43°C reference for CEM43
            spatial_order: 4,
        }
    }
}

/// Thermal diffusion plugin for the physics system
#[derive(Debug)]
pub struct ThermalDiffusionPlugin {
    metadata: PluginMetadata,
    solver: Option<ThermalDiffusionSolver>,
    config: ThermalDiffusionConfig,
    state: PluginState,
}

impl ThermalDiffusionPlugin {
    #[must_use]
    pub fn new(config: ThermalDiffusionConfig) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "thermal_diffusion".to_string(),
                name: "ThermalDiffusion".to_string(),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: "Thermal diffusion solver with bioheat and hyperbolic models"
                    .to_string(),
                license: "MIT".to_string(),
            },
            solver: None,
            config,
            state: PluginState::Created,
        }
    }

    #[allow(dead_code)]
    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl crate::physics::plugin::Plugin for ThermalDiffusionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        // Temperature solver may need heat source from acoustic absorption
        vec![] // Optional: heat_source field
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Temperature]
    }

    fn initialize(&mut self, grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.solver = Some(ThermalDiffusionSolver::new(self.config.clone(), grid));
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        if let Some(ref mut solver) = self.solver {
            // Get heat source from fields if available (could be from acoustic absorption)
            let heat_source = if fields.shape()[0] > UnifiedFieldType::Temperature as usize + 1 {
                Some(
                    fields
                        .index_axis(ndarray::Axis(0), UnifiedFieldType::Temperature as usize + 1)
                        .to_owned(),
                )
            } else {
                None
            };

            solver.update(medium, grid, dt, heat_source.as_ref())?;

            // Store temperature back in fields
            let temp_idx = UnifiedFieldType::Temperature as usize;
            if fields.shape()[0] > temp_idx {
                let mut temp_field = fields.index_axis_mut(ndarray::Axis(0), temp_idx);
                temp_field.assign(solver.temperature());
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;
    use ndarray::Array3;

    #[test]
    fn test_thermal_diffusion_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = ThermalDiffusionConfig::default();
        let solver = ThermalDiffusionSolver::new(config, &grid);
        assert_eq!(solver.temperature().shape(), &[32, 32, 32]);
    }

    #[test]
    fn test_heat_diffusion() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);

        let config = ThermalDiffusionConfig {
            enable_bioheat: false,
            enable_hyperbolic: false,
            ..Default::default()
        };

        let mut solver = ThermalDiffusionSolver::new(config, &grid);

        // Set initial hot spot
        let mut initial_temp = Array3::from_elem((16, 16, 16), 310.0); // Body temperature
        initial_temp[[8, 8, 8]] = 320.0; // Hot spot
        solver.set_temperature(initial_temp);

        // Run diffusion
        for _ in 0..10 {
            solver.update(&medium, &grid, 0.001, None).unwrap();
        }

        // Check that heat has diffused
        let final_temp = solver.temperature();
        assert!(final_temp[[8, 8, 8]] < 320.0); // Center cooled
        assert!(final_temp[[7, 8, 8]] > 310.0); // Neighbors heated
    }
}
