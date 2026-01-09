//! FDTD solver plugin implementation

use ndarray::Array4;
use std::fmt::Debug;

use super::{FdtdConfig, FdtdSolver};
use crate::core::error::{KwaversError, KwaversResult, PhysicsError};
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::GridSource;
use crate::physics::plugin::{PluginContext, PluginMetadata, PluginState};

/// FDTD solver plugin
#[derive(Debug)]
pub struct FdtdPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    config: FdtdConfig,
    solver: Option<FdtdSolver>,
}

impl FdtdPlugin {
    /// Create a new FDTD plugin
    pub fn new(config: FdtdConfig, _grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            metadata: PluginMetadata {
                id: "fdtd_solver".to_string(),
                name: "FDTD Solver".to_string(),
                version: "1.0.0".to_string(),
                description: "Finite-Difference Time Domain solver for acoustic wave propagation"
                    .to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            config,
            solver: None,
        })
    }

    #[allow(dead_code)]
    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    #[allow(dead_code)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[allow(dead_code)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl crate::physics::plugin::Plugin for FdtdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
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
        let source = GridSource::default(); // Default source (no active sources unless configured elsewhere)

        let solver = FdtdSolver::new(self.config.clone(), grid, medium, source)?;

        self.solver = Some(solver);
        self.state = PluginState::Running;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        dt: f64,
        _t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        // Ensure fields have correct dimensions
        if fields.dim().0 <= UnifiedFieldType::VelocityZ.index() {
            return Err(crate::core::error::KwaversError::Physics(
                crate::core::error::PhysicsError::InvalidFieldDimensions {
                    expected: "pressure + 3 velocity components".to_string(),
                    actual: format!("{} components", fields.dim().0),
                },
            ));
        }

        let solver = self.solver.as_mut().ok_or_else(|| {
            crate::core::error::KwaversError::Physics(
                crate::core::error::PhysicsError::ModelNotInitialized {
                    model: "FDTD Solver".to_string(),
                    reason: "Solver not initialized by plugin lifecycle".to_string(),
                },
            )
        })?;

        let max_sound_speed = solver
            .materials
            .c0
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x));
        let max_dt = solver.max_stable_dt(max_sound_speed);
        if dt > max_dt {
            return Err(KwaversError::Physics(PhysicsError::NumericalInstability {
                timestep: dt,
                cfl_limit: max_dt,
            }));
        }

        solver.config.dt = dt;

        // Extract field slices
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        // Sync input fields to solver state
        // Note: Solver owns its state (p, ux, uy, uz). We overwrite it with input fields.
        // This allows the plugin chain to modify fields before FDTD step.
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

        // Perform time step
        solver.step_forward()?;

        context.boundary.apply_acoustic(
            solver.fields.p.view_mut(),
            grid,
            solver.time_step_index,
        )?;

        // Sync output fields from solver state
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

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn diagnostics(&self) -> String {
        format!("FDTD Plugin - State: {:?}", self.state)
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
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
