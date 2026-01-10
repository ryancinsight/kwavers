//! Spectral solver plugin implementation

use ndarray::Array4;
use std::fmt::Debug;

use super::{PSTDConfig, PSTDSolver, PSTDSource};
use crate::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::plugin::{PluginContext, PluginMetadata, PluginState};

/// PSTD solver plugin
#[derive(Debug)]
pub struct PSTDPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: Option<PSTDSolver>,
    config: PSTDConfig,
}

impl PSTDPlugin {
    /// Create a new PSTD plugin
    pub fn new(config: PSTDConfig, _grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            metadata: PluginMetadata {
                id: "pstd_solver".to_string(),
                name: "PSTD Solver".to_string(),
                version: "1.0.0".to_string(),
                description: "Generalized spectral solver for acoustic wave propagation"
                    .to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Created,
            solver: None,
            config,
        })
    }
}

impl crate::domain::plugin::Plugin for PSTDPlugin {
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
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
            UnifiedFieldType::Density,
            UnifiedFieldType::SoundSpeed,
        ]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
        ]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let source = PSTDSource::default();
        let solver = PSTDSolver::new(self.config.clone(), grid.clone(), medium, source)?;
        self.solver = Some(solver);
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
            crate::core::error::KwaversError::InternalError(
                "PSTD solver not initialized".to_string(),
            )
        })?;

        // Sync from global fields to internal solver state
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        solver
            .fields
            .p
            .assign(&fields.index_axis(ndarray::Axis(0), pressure_idx));
        solver
            .fields
            .ux
            .assign(&fields.index_axis(ndarray::Axis(0), vx_idx));
        solver
            .fields
            .uy
            .assign(&fields.index_axis(ndarray::Axis(0), vy_idx));
        solver
            .fields
            .uz
            .assign(&fields.index_axis(ndarray::Axis(0), vz_idx));

        // Sync density from pressure (Linear EOS approximation) to ensure consistency
        // rho = p / c0^2
        ndarray::Zip::from(&mut solver.rho)
            .and(&solver.fields.p)
            .and(&solver.materials.c0)
            .for_each(|rho, &p, &c| {
                if c > 1e-6 {
                    *rho = p / (c * c);
                } else {
                    *rho = 0.0;
                }
            });

        // Perform time step
        solver.step_forward()?;

        // Sync back to global fields
        fields
            .index_axis_mut(ndarray::Axis(0), pressure_idx)
            .assign(&solver.fields.p);
        fields
            .index_axis_mut(ndarray::Axis(0), vx_idx)
            .assign(&solver.fields.ux);
        fields
            .index_axis_mut(ndarray::Axis(0), vy_idx)
            .assign(&solver.fields.uy);
        fields
            .index_axis_mut(ndarray::Axis(0), vz_idx)
            .assign(&solver.fields.uz);

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
