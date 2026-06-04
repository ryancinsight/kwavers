//! Spectral solver plugin implementation

use ndarray::Array4;
use std::fmt::Debug;

use super::{PSTDConfig, PSTDSolver};
use kwavers_core::error::KwaversResult;
use kwavers_domain::field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_domain::plugin::{PluginContext, PluginMetadata, PluginState};
use kwavers_domain::source::GridSource;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: PSTDConfig, _grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            metadata: PluginMetadata {
                id: "pstd_solver".to_owned(),
                name: "PSTD Solver".to_owned(),
                version: "1.0.0".to_owned(),
                description: "Generalized spectral solver for acoustic wave propagation".to_owned(),
                author: "Kwavers Team".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            solver: None,
            config,
        })
    }
}

impl kwavers_domain::plugin::Plugin for PSTDPlugin {
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
        let source = GridSource::default();
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
            kwavers_core::error::KwaversError::InternalError(
                "PSTD solver not initialized".to_owned(),
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

        // Sync density from pressure (Linear EOS approximation) to ensure consistency.
        //
        // Not yet implemented: higher-order PSTD methods. Absent: 4th-order k-space
        // derivative schemes; Tait or stiffened-gas nonlinear equation of state for
        // shock formation; multi-scale inter-band frequency coupling; anisotropic
        // dispersion correction for broadband pulses; PML absorbing boundaries; and
        // GPU acceleration for large-scale 3D domains.
        // rho = p / c0^2, split across rhox/rhoy/rhoz
        ndarray::Zip::from(&mut solver.rhox)
            .and(&mut solver.rhoy)
            .and(&mut solver.rhoz)
            .and(&solver.fields.p)
            .and(&solver.materials.c0)
            .par_for_each(|rx, ry, rz, &p, &c| {
                if c > 1e-6 {
                    let rho = p / (c * c);
                    let split = rho / 3.0;
                    *rx = split;
                    *ry = split;
                    *rz = split;
                } else {
                    *rx = 0.0;
                    *ry = 0.0;
                    *rz = 0.0;
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
