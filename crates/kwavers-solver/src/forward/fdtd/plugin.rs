//! FDTD solver plugin implementation

use leto::Array4;
use std::fmt::Debug;

use super::{FdtdConfig, FdtdSolver};
use crate::plugin::{PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_source::GridSource;

fn copy_ndarray_view_into_leto(dst: &mut leto::Array3<f64>, src: leto::ArrayView3<'_, f64>) {
    for (dst_value, src_value) in dst
        .as_slice_mut()
        .expect("leto FDTD field must be contiguous")
        .iter_mut()
        .zip(src.iter())
    {
        *dst_value = *src_value;
    }
}

fn copy_leto_into_ndarray(dst: &mut leto::ArrayViewMut3<'_, f64>, src: &leto::Array3<f64>) {
    for (dst_value, src_value) in dst.iter_mut().zip(src.iter()) {
        *dst_value = *src_value;
    }
}

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: FdtdConfig, _grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            metadata: PluginMetadata {
                id: "fdtd_solver".to_owned(),
                name: "FDTD Solver".to_owned(),
                version: "1.0.0".to_owned(),
                description: "Finite-Difference Time Domain solver for acoustic wave propagation"
                    .to_owned(),
                author: "Kwavers Team".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Initialized,
            config,
            solver: None,
        })
    }
}

impl crate::plugin::Plugin for FdtdPlugin {
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
        if fields.shape()[0] <= UnifiedFieldType::VelocityZ.index() {
            return Err(kwavers_core::error::KwaversError::Physics(
                kwavers_core::error::PhysicsError::InvalidFieldDimensions {
                    expected: "pressure + 3 velocity components".to_owned(),
                    actual: format!("{} components", fields.shape()[0]),
                },
            ));
        }

        let solver = self.solver.as_mut().ok_or_else(|| {
            kwavers_core::error::KwaversError::Physics(
                kwavers_core::error::PhysicsError::ModelNotInitialized {
                    model: "FDTD Solver".to_owned(),
                    reason: "Solver not initialized by plugin lifecycle".to_owned(),
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
        copy_ndarray_view_into_leto(
            &mut solver.fields.p,
            fields.index_axis(0, pressure_idx),
        );
        copy_ndarray_view_into_leto(
            &mut solver.fields.ux,
            fields.index_axis(0, vx_idx),
        );
        copy_ndarray_view_into_leto(
            &mut solver.fields.uy,
            fields.index_axis(0, vy_idx),
        );
        copy_ndarray_view_into_leto(
            &mut solver.fields.uz,
            fields.index_axis(0, vz_idx),
        );

        // Perform time step
        solver.step_forward()?;

        let [nx, ny, nz] = solver.fields.p.shape();
        let mut pressure = leto::Array3::from_shape_vec(
            (nx, ny, nz),
            solver.fields.p.iter().copied().collect(),
        )
        .expect("leto pressure field shape must map to ndarray");
        context
            .boundary
            .apply_acoustic(pressure.view_mut().unwrap().into(), grid, solver.time_step_index)?;
        for (dst_value, src_value) in solver
            .fields
            .p
            .as_slice_mut()
            .expect("leto FDTD pressure field must be contiguous")
            .iter_mut()
            .zip(pressure.iter())
        {
            *dst_value = *src_value;
        }

        // Sync output fields from solver state
        copy_leto_into_ndarray(
            &mut fields.index_axis_mut(0, pressure_idx),
            &solver.fields.p,
        );
        copy_leto_into_ndarray(
            &mut fields.index_axis_mut(0, vx_idx),
            &solver.fields.ux,
        );
        copy_leto_into_ndarray(
            &mut fields.index_axis_mut(0, vy_idx),
            &solver.fields.uy,
        );
        copy_leto_into_ndarray(
            &mut fields.index_axis_mut(0, vz_idx),
            &solver.fields.uz,
        );

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
