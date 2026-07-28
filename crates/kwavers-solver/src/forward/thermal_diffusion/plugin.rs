use crate::plugin::{PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::thermal::{VolumetricHeatSource, diffusion::ThermalDiffusionConfig};
use leto::Array4;

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
                id: "thermal_diffusion".to_owned(),
                name: "ThermalDiffusion".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Thermal diffusion solver with bioheat and hyperbolic models"
                    .to_owned(),
                license: "MIT".to_owned(),
            },
            solver: None,
            config,
            state: PluginState::Created,
        }
    }
}

impl crate::plugin::Plugin for ThermalDiffusionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::VolumetricHeatSource]
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
            let source_field = fields
                .index_axis::<3>(0, UnifiedFieldType::VolumetricHeatSource.index())
                .map_err(|_| {
                    KwaversError::DimensionMismatch(format!(
                        "thermal diffusion plugin requires field axis index {} but fields has {} axes",
                        UnifiedFieldType::VolumetricHeatSource.index(),
                        fields.shape()[0]
                    ))
                })?;

            // Keep the borrowed field view inside this scope so the mutable
            // temperature write below cannot alias the deposition input.
            solver.update(
                medium,
                grid,
                dt,
                Some(VolumetricHeatSource::from_base(source_field)),
            )?;

            let field_count = fields.shape()[0];
            let mut temp_field = fields
                .index_axis_mut::<3>(0, UnifiedFieldType::Temperature.index())
                .map_err(|_| {
                    KwaversError::DimensionMismatch(format!(
                        "thermal diffusion plugin requires field axis index {} but fields has {} axes",
                        UnifiedFieldType::Temperature.index(),
                        field_count
                    ))
                })?;
            temp_field.assign(solver.temperature());
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
    use crate::plugin::Plugin;
    use crate::plugin::test_support::{NullBoundary, make_context, null_plugin_fields};
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
    use kwavers_medium::{HomogeneousMedium, ThermalProperties};
    use leto::{Array3, Array4};

    #[test]
    fn test_thermal_diffusion_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = ThermalDiffusionConfig::default();
        let solver = ThermalDiffusionSolver::new(config, &grid);
        assert_eq!(solver.temperature().shape(), [32, 32, 32]);
    }

    #[test]
    fn test_heat_diffusion() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);

        let config = ThermalDiffusionConfig {
            enable_bioheat: false,
            enable_hyperbolic: false,
            ..Default::default()
        };

        let mut solver = ThermalDiffusionSolver::new(config, &grid);

        let mut initial_temp = Array3::from_elem([16, 16, 16], BODY_TEMPERATURE_K);
        initial_temp[[8, 8, 8]] = 320.0;
        solver.set_temperature(initial_temp);

        for _ in 0..10 {
            solver.update(&medium, &grid, 0.001, None).unwrap();
        }

        let final_temp = solver.temperature();
        assert!(final_temp[[8, 8, 8]] < 320.0);
        assert!(final_temp[[7, 8, 8]] > BODY_TEMPERATURE_K);
    }

    #[test]
    fn plugin_consumes_volumetric_heat_source_field() {
        let grid = Grid::new(3, 3, 1, 1.0, 1.0, 1.0).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
        let config = ThermalDiffusionConfig {
            arterial_temperature: 310.0,
            enable_bioheat: false,
            enable_hyperbolic: false,
            ..Default::default()
        };
        let mut plugin = ThermalDiffusionPlugin::new(config);
        plugin.initialize(&grid, &medium).unwrap();

        assert_eq!(
            plugin.required_fields(),
            vec![UnifiedFieldType::VolumetricHeatSource]
        );

        let mut fields = Array4::zeros((UnifiedFieldType::COUNT, grid.nx, grid.ny, grid.nz));
        let rho = kwavers_medium::density_at(&medium, 1.0, 1.0, 0.0, &grid);
        let cp = medium.specific_heat(1.0, 1.0, 0.0, &grid);
        let heating_rate = 5.0;
        fields[[UnifiedFieldType::VolumetricHeatSource.index(), 1, 1, 0]] = heating_rate * rho * cp;

        let extra_fields = null_plugin_fields(&grid);
        let mut boundary = NullBoundary;
        let mut context = make_context(&extra_fields, &mut boundary);
        plugin
            .update(&mut fields, &grid, &medium, 2.0, 0.0, &mut context)
            .unwrap();

        let expected = 310.0 + heating_rate * 2.0;
        let actual = fields[[UnifiedFieldType::Temperature.index(), 1, 1, 0]];
        let tolerance = expected.abs() * 4.0 * f64::EPSILON;
        assert!((actual - expected).abs() <= tolerance);
        assert_eq!(
            fields[[UnifiedFieldType::Temperature.index(), 0, 0, 0]],
            310.0
        );
    }
}
