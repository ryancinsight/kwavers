use crate::domain::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::plugin::{PluginContext, PluginMetadata, PluginState};
use crate::physics::thermal::diffusion::ThermalDiffusionConfig;
use ndarray::Array4;

use super::solver::ThermalDiffusionSolver;

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
}

impl crate::domain::plugin::Plugin for ThermalDiffusionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![]
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
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        if let Some(ref mut solver) = self.solver {
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
    use crate::domain::medium::HomogeneousMedium;
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

        let mut initial_temp = Array3::from_elem((16, 16, 16), 310.0);
        initial_temp[[8, 8, 8]] = 320.0;
        solver.set_temperature(initial_temp);

        for _ in 0..10 {
            solver.update(&medium, &grid, 0.001, None).unwrap();
        }

        let final_temp = solver.temperature();
        assert!(final_temp[[8, 8, 8]] < 320.0);
        assert!(final_temp[[7, 8, 8]] > 310.0);
    }
}
