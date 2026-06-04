//! `Plugin` trait implementation for `KzkSolverPlugin`.

use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_domain::plugin::{PluginMetadata, PluginState};

use super::solver::KzkSolverPlugin;

impl kwavers_domain::plugin::Plugin for KzkSolverPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<kwavers_field::mapping::UnifiedFieldType> {
        vec![kwavers_field::mapping::UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<kwavers_field::mapping::UnifiedFieldType> {
        vec![kwavers_field::mapping::UnifiedFieldType::Pressure]
    }

    fn update(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &mut kwavers_domain::plugin::PluginContext<'_>,
    ) -> KwaversResult<()> {
        use kwavers_field::mapping::UnifiedFieldType;

        let pressure_field =
            fields.index_axis(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        let mut pressure_array = pressure_field.to_owned();

        if let Some(operators) = &self.frequency_operators {
            self.apply_linear_step(&mut pressure_array, operators, dt / 2.0)?;

            let density = kwavers_medium::density_at(medium, 0.0, 0.0, 0.0, grid);
            let c0 = kwavers_medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
            let beta = kwavers_medium::AcousticProperties::nonlinearity_coefficient(
                medium, 0.0, 0.0, 0.0, grid,
            );

            self.apply_nonlinear_step(&mut pressure_array, beta, density, c0, dt, grid)?;
            self.apply_linear_step(&mut pressure_array, operators, dt / 2.0)?;

            let mut pressure_slice =
                fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
            pressure_slice.assign(&pressure_array);
        }

        Ok(())
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let default_freq = MHZ_TO_HZ;
        self.initialize_operators(grid, medium, default_freq)?;
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.frequency_operators = None;
        self.retarded_time_window = None;
        self.state = PluginState::Created;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
