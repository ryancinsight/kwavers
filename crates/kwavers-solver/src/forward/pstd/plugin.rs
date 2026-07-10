//! Spectral solver plugin implementation

use leto::Array4;
use std::fmt::Debug;

use super::{PSTDConfig, PSTDSolver};
use crate::plugin::{PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::KwaversResult;
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_source::GridSource;

fn copy_ndarray_view_into_leto(dst: &mut leto::Array3<f64>, src: leto::ArrayView3<'_, f64>) {
    for (dst_value, src_value) in dst
        .as_slice_mut()
        .expect("leto PSTD field must be contiguous")
        .iter_mut()
        .zip(src.iter())
    {
        *dst_value = *src_value;
    }
}

fn copy_leto_into_ndarray(dst: &mut leto::ArrayViewMut3<'_, f64>, src: &leto::Array3<f64>) {
    leto_ops::zip_mut_with(dst, &src.view(), |d, s| *d = *s)
        .expect("invariant: dst and src share shape");
}

/// PSTD solver plugin
#[derive(Debug)]
pub struct PSTDPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: Option<PSTDSolver>,
    config: PSTDConfig,
    /// Whether the zero-velocity IVP has been seeded from the externally-injected
    /// initial pressure (done once, on the first `update`).
    ivp_seeded: bool,
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
            ivp_seeded: false,
        })
    }
}

impl crate::plugin::Plugin for PSTDPlugin {
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
        self.ivp_seeded = false;
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

        copy_ndarray_view_into_leto(
            &mut solver.fields.p,
            fields
                .index_axis::<3>(0, pressure_idx)
                .expect("invariant: pressure field index within unified field array"),
        );

        if self.ivp_seeded {
            // Steady state: velocity is genuine evolving state, sync it in. The
            // partial densities (ρx, ρy, ρz) are the solver's own leapfrog state
            // and MUST persist across steps — they are NOT re-derived from the
            // pressure here. Re-splitting ρ = p/c² equally every step destroys the
            // directional density information and collapses the wave amplitude
            // (the historical bug this replaces).
            copy_ndarray_view_into_leto(
                &mut solver.fields.ux,
                fields
                    .index_axis::<3>(0, vx_idx)
                    .expect("invariant: velocity-x field index within unified field array"),
            );
            copy_ndarray_view_into_leto(
                &mut solver.fields.uy,
                fields
                    .index_axis::<3>(0, vy_idx)
                    .expect("invariant: velocity-y field index within unified field array"),
            );
            copy_ndarray_view_into_leto(
                &mut solver.fields.uz,
                fields
                    .index_axis::<3>(0, vz_idx)
                    .expect("invariant: velocity-z field index within unified field array"),
            );
        } else {
            // First step: the field array carries the zero-velocity IVP initial
            // pressure (`fields.p` just synced). PSTD's state is the partial
            // densities with pressure derived via the EOS, so seed both the
            // densities (ρ = p₀/c², split over active dims) and the half-step
            // velocity. The incoming velocity is zero by construction, so it is not
            // synced over the seeded half-step velocity.
            solver.seed_ivp_from_pressure()?;
            self.ivp_seeded = true;
        }

        // Perform time step
        solver.step_forward()?;

        // Sync back to global fields
        copy_leto_into_ndarray(
            &mut fields
                .index_axis_mut::<3>(0, pressure_idx)
                .expect("invariant: pressure field index within unified field array"),
            &solver.fields.p,
        );
        copy_leto_into_ndarray(
            &mut fields
                .index_axis_mut::<3>(0, vx_idx)
                .expect("invariant: velocity-x field index within unified field array"),
            &solver.fields.ux,
        );
        copy_leto_into_ndarray(
            &mut fields
                .index_axis_mut::<3>(0, vy_idx)
                .expect("invariant: velocity-y field index within unified field array"),
            &solver.fields.uy,
        );
        copy_leto_into_ndarray(
            &mut fields
                .index_axis_mut::<3>(0, vz_idx)
                .expect("invariant: velocity-z field index within unified field array"),
            &solver.fields.uz,
        );

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
