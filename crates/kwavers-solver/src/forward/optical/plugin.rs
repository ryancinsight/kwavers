//! Plugin adapter for the optical diffusion solver.

use std::any::Any;
use crate::forward::optical::solver::OpticalDiffusionSolver;
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::{Array3, Array4};
#[derive(Debug)]
pub struct OpticalDiffusionPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: Option<OpticalDiffusionSolver>,
    initialized_source: bool,
}

impl Default for OpticalDiffusionPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl OpticalDiffusionPlugin {
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "optical_diffusion".to_owned(),
                name: "Optical Diffusion".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Steady-state optical diffusion for photoacoustic source generation"
                    .to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            solver: None,
            initialized_source: false,
        }
    }
}

impl Plugin for OpticalDiffusionPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        Vec::new()
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::LightFluence]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let mut solver = OpticalDiffusionSolver::from_medium(grid.clone(), medium)?;
        let source = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
        let _ = solver.solve(&source)?;
        self.solver = Some(solver);
        self.initialized_source = true;
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let solver = self.solver.as_mut().ok_or_else(|| {
            KwaversError::InternalError("OpticalDiffusionPlugin updated before initialize()".to_owned())
        })?;

        let light_index = UnifiedFieldType::LightFluence.index();
        let pressure_index = UnifiedFieldType::Pressure.index();
        let source_view = fields
            .index_axis::<3>(0, light_index)
            .expect("invariant: light fluence field exists");

        let has_external_source = source_view.iter().any(|value| value.abs() > 0.0);
        if has_external_source || !self.initialized_source {
            let source = source_view.to_contiguous();
            let _ = solver.solve(&source)?;
            self.initialized_source = true;
        }

        fields
            .index_axis_mut::<3>(0, light_index)
            .expect("invariant: light fluence field exists")
            .assign(&solver.fluence());
        fields
            .index_axis_mut::<3>(0, pressure_index)
            .expect("invariant: pressure field exists")
            .assign(&solver.absorbed_energy_density());

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
