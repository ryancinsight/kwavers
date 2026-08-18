//! Plugin adapter wrapping the correct complex-field [`KZKSolver`] for the
//! [`Plugin`](crate::plugin::Plugin) trait system.
//!
//! The [`KZKSolver`] is a parabolic beam-propagation solver that advances
//! through z-planes using Strang operator splitting.  This adapter runs the
//! full z-propagation on the first [`Plugin::update`] call and serves cached
//! z-slices thereafter, matching the therapy `execution.rs` pattern where the
//! entire volume is computed before per-plane readout.
//!
//! # Coordinate mapping
//!
//! The therapy frame uses axial = x (`grid.nx`), transverse = (y, z).
//! The KZK solver uses axial = z, transverse = (x, y).  The adapter remaps:
//!
//! | KZK field | Therapy field |
//! |-----------|---------------|
//! | `nx`      | `grid.ny`     |
//! | `ny`      | `grid.nz`     |
//! | `nz`      | `grid.nx`     |
//! | `dx`      | `grid.dy`     |
//! | `dz`      | `grid.dx`     |

use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::wave_propagation::KZKSolverTrait;
use leto::{Array2, Array3, Array4};

use super::solver::KZKSolver;
use super::KZKConfig;
use crate::plugin::{PluginMetadata, PluginState};

/// Plugin adapter for the correct complex-field KZK solver.
///
/// Wraps [`KZKSolver`] behind the [`crate::plugin::Plugin`] trait so the
/// [`PhysicsCatalog`](crate::plugin::catalog::PhysicsCatalog) can instantiate it.
pub struct KzkPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: Option<KZKSolver>,
    config: Option<KZKConfig>,
    /// Cached RMS volume from full propagation, shape `(grid.ny, grid.nz, grid.nx)`.
    cached_volume: Option<Array3<f64>>,
    /// Current z-slice index into the cached volume.
    current_step: usize,
}

impl std::fmt::Debug for KzkPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KzkPlugin")
            .field("metadata", &self.metadata)
            .field("state", &self.state)
            .field("has_solver", &self.solver.is_some())
            .field("has_volume", &self.cached_volume.is_some())
            .field("current_step", &self.current_step)
            .finish()
    }
}

// SAFETY: KZKSolver owns all its data (Array3<Complex64>, FFTW plans,
// operators).  All inner types are Send + Sync — complex arrays, real
// arrays, and FFTW forward/inverse plan wrappers that are safe to share
// across threads.  KZKSolver contains no raw pointers or thread-local
// state.
unsafe impl Send for KzkPlugin {}
unsafe impl Sync for KzkPlugin {}

impl Default for KzkPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl KzkPlugin {
    /// Create new KZK plugin adapter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "kzk_solver".to_owned(),
                name: "KZK Beam Propagation Solver".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Complex-field KZK beam propagation using Strang splitting".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            solver: None,
            config: None,
            cached_volume: None,
            current_step: 0,
        }
    }

    /// Build a [`KZKConfig`] from grid and medium parameters.
    ///
    /// Applies the coordinate mapping: KZK axial = therapy x, KZK transverse =
    /// therapy (y, z).
    fn build_config(grid: &Grid, medium: &dyn Medium) -> KZKConfig {
        let c0 = kwavers_medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        let rho0 = kwavers_medium::density_at(medium, 0.0, 0.0, 0.0, grid);
        let b_over_a = medium.nonlinearity_parameter(0.0, 0.0, 0.0, grid);
        let alpha0 = medium.alpha_coefficient(0.0, 0.0, 0.0, grid);
        let alpha_power = medium.alpha_power(0.0, 0.0, 0.0, grid);

        // CFL constraint: c0 * dt / dz <= 0.5 for the parabolic approximation.
        let dz = grid.dx; // therapy axial spacing → KZK axial step
        let mut dt = 0.4 * dz / c0; // conservative CFL = 0.4

        // Nyquist: dt <= 1 / (20 * f_max), where f_max ≈ 10 * fundamental.
        let frequency = 1.0e6; // 1 MHz default
        let f_max = 10.0 * frequency;
        let dt_nyquist = 1.0 / (20.0 * f_max);
        if dt > dt_nyquist {
            dt = dt_nyquist;
        }

        // Number of retarded-time samples: enough to resolve the waveform.
        let period = 1.0 / frequency;
        let mut nt = (period / dt).ceil() as usize;
        if nt < 64 {
            nt = 64; // minimum for spectral accuracy
        }

        KZKConfig {
            // Coordinate mapping: KZK nx,ny = therapy ny,nz (transverse);
            // KZK nz = therapy nx (axial).
            nx: grid.ny,
            ny: grid.nz,
            nz: grid.nx,
            dx: grid.dy,
            dz,
            dt,
            nt,
            c0,
            rho0,
            b_over_a,
            alpha0,
            alpha_power,
            include_diffraction: true,
            include_absorption: true,
            include_nonlinearity: true,
            frequency,
        }
    }

    /// Extract a 2D source amplitude map from the current 3D pressure field.
    ///
    /// Collapses the therapy axial (x) dimension by summing absolute values,
    /// producing a `(grid.ny, grid.nz)` amplitude map that maps to KZK
    /// transverse coordinates.
    fn extract_source(fields: &Array4<f64>, grid: &Grid) -> Array2<f64> {
        let kzk_nx = grid.ny;
        let kzk_ny = grid.nz;
        let mut source = Array2::<f64>::zeros((kzk_nx, kzk_ny));

        let pressure_field = fields
            .index_axis::<3>(0, UnifiedFieldType::Pressure.index())
            .expect("invariant: pressure field index within field stack");

        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let mut amp = 0.0_f64;
                for ix in 0..grid.nx {
                    amp += pressure_field[[ix, iy, iz]].abs();
                }
                source[[iy, iz]] = amp;
            }
        }

        // Normalise so the peak amplitude is 1 Pa (unit source).
        let peak = source.iter().cloned().fold(0.0_f64, f64::max);
        if peak > 0.0 {
            for v in source.iter_mut() {
                *v /= peak;
            }
        }

        source
    }
}

impl crate::plugin::Plugin for KzkPlugin {
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
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let config = Self::build_config(grid, medium);

        let solver = KZKSolver::new(config.clone())
            .map_err(|msg| KwaversError::InternalError(format!("KZKSolver::new failed: {msg}")))?;

        self.solver = Some(solver);
        self.config = Some(config);
        self.cached_volume = None;
        self.current_step = 0;
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &mut crate::plugin::PluginContext<'_>,
    ) -> KwaversResult<()> {
        // On first call: run full z-propagation and cache the RMS volume.
        if self.cached_volume.is_none() {
            let solver = self.solver.as_mut().ok_or_else(|| {
                KwaversError::InternalError("KzkPlugin::update called before initialize".to_owned())
            })?;

            let source = Self::extract_source(fields, grid);
            solver.set_source(source, 1.0e6);

            // Collect RMS at each z-plane → volume shape (kzk_nx, kzk_ny, nz).
            let kzk_nx = solver.config.nx;
            let kzk_ny = solver.config.ny;
            let total_z = solver.config.nz;
            let mut volume = Array3::<f64>::zeros((kzk_nx, kzk_ny, total_z));

            for iz in 0..total_z {
                solver.step();
                let rms_2d = solver.current_field(); // (kzk_nx, kzk_ny)
                for i in 0..kzk_nx {
                    for j in 0..kzk_ny {
                        volume[[i, j, iz]] = rms_2d[[i, j]];
                    }
                }
            }

            self.cached_volume = Some(volume);
        }

        // Write the current z-slice back into the pressure field.
        if let Some(ref volume) = self.cached_volume {
            let total_z = volume.shape()[2];
            let step = self.current_step.min(total_z.saturating_sub(1));
            let kzk_nx = volume.shape()[0];
            let kzk_ny = volume.shape()[1];

            let mut pressure_slice = fields
                .index_axis_mut::<3>(0, UnifiedFieldType::Pressure.index())
                .expect("invariant: pressure field index within field stack");

            // Write RMS at the current axial plane (therapy x = step).
            let therapy_x = step.min(grid.nx.saturating_sub(1));
            for iy in 0..grid.ny.min(kzk_nx) {
                for iz in 0..grid.nz.min(kzk_ny) {
                    pressure_slice[[therapy_x, iy, iz]] = volume[[iy, iz, step]];
                }
            }

            if self.current_step < total_z {
                self.current_step += 1;
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.solver = None;
        self.config = None;
        self.cached_volume = None;
        self.current_step = 0;
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.solver = None;
        self.config = None;
        self.cached_volume = None;
        self.current_step = 0;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::test_support::{make_context, null_plugin_fields, NullBoundary};
    use crate::plugin::Plugin;
    use kwavers_core::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use leto::Array4;

    fn small_grid() -> Grid {
        // Axial-elongated: grid.nx (therapy axial) > grid.ny so the KZK
        // parabolic angle check (atan(KZK_nx·dx / (2·KZK_nz·dz))) passes.
        // KZK_nx = grid.ny = 16, KZK_nz = grid.nx = 32 → atan(16/64) ≈ 14°.
        Grid::new(32, 16, 16, 1e-3, 1e-3, 1e-3).expect("grid")
    }

    fn water(grid: &Grid) -> HomogeneousMedium {
        HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, grid)
    }

    /// Plane-wave absorption oracle: lossless medium preserves finite,
    /// positive amplitude through the propagation.
    #[test]
    fn plane_wave_absorption_oracle() {
        let grid = small_grid();
        let mut medium = water(&grid);
        medium
            .set_acoustic_properties(0.0, 1.0, 5.0)
            .expect("set_acoustic_properties");

        let mut plugin = KzkPlugin::new();
        plugin.initialize(&grid, &medium).expect("initialize");

        // Uniform plane wave of 1 Pa.
        let mut fields = Array4::<f64>::zeros((1, grid.nx, grid.ny, grid.nz));
        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                for iz in 0..grid.nz {
                    fields[[0, ix, iy, iz]] = 1.0;
                }
            }
        }

        let extra = null_plugin_fields(&grid);
        let mut boundary = NullBoundary;
        let mut ctx = make_context(&extra, &mut boundary);
        let dt = 1.0e-7;

        // First update triggers full propagation and writes the first slice.
        plugin
            .update(&mut fields, &grid, &medium, dt, 0.0, &mut ctx)
            .expect("update");

        // The initial slice must be finite and positive.
        let p000 = fields[[0, 0, 0, 0]];
        assert!(
            p000.is_finite() && p000 > 0.0,
            "lossless plane-wave: p(0,0,0) must be finite positive, got {p000}"
        );
    }

    /// Plugin must evolve the field: a centred pulse in lossless water
    /// changes after update steps.
    #[test]
    fn plugin_evolves_real_field() {
        let grid = small_grid();
        let medium = water(&grid);
        let mut plugin = KzkPlugin::new();
        plugin.initialize(&grid, &medium).expect("initialize");

        let mut fields = Array4::<f64>::zeros((1, grid.nx, grid.ny, grid.nz));
        fields[[0, 8, 8, 8]] = 1.0e5;
        let before = fields.clone();

        let extra = null_plugin_fields(&grid);
        let mut boundary = NullBoundary;
        let mut ctx = make_context(&extra, &mut boundary);
        let dt = 5.0e-8;

        for step in 0..5 {
            plugin
                .update(&mut fields, &grid, &medium, dt, step as f64 * dt, &mut ctx)
                .expect("update");
        }

        assert!(
            fields.iter().all(|v| v.is_finite()),
            "field must remain finite (stable step)"
        );
        let max_change = fields
            .iter()
            .zip(before.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_change > 0.0,
            "plugin must evolve the field (real computation); max change = {max_change} Pa"
        );
    }

    /// Focused beam: Gaussian source propagates and produces non-zero
    /// on-axis amplitude at the midpoint.
    #[test]
    fn focused_beam_amplitude() {
        let grid = Grid::new(32, 16, 16, 0.5e-3, 0.5e-3, 0.5e-3).expect("grid");
        let medium = water(&grid);
        let mut plugin = KzkPlugin::new();
        plugin.initialize(&grid, &medium).expect("initialize");

        let mut fields = Array4::<f64>::zeros((1, grid.nx, grid.ny, grid.nz));
        // Gaussian source centred on the transverse plane (iy, iz).
        let cy = grid.ny as f64 / 2.0;
        let cz = grid.nz as f64 / 2.0;
        let sigma = grid.ny as f64 / 4.0;
        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                for iz in 0..grid.nz {
                    let r2 =
                        ((iy as f64 - cy).powi(2) + (iz as f64 - cz).powi(2)) / (sigma * sigma);
                    fields[[0, ix, iy, iz]] = (-r2).exp();
                }
            }
        }

        let extra = null_plugin_fields(&grid);
        let mut boundary = NullBoundary;
        let mut ctx = make_context(&extra, &mut boundary);
        let dt = 5.0e-8;

        // Run through all z-slices (grid.nx = therapy axial = KZK nz).
        for step in 0..grid.nx {
            plugin
                .update(&mut fields, &grid, &medium, dt, step as f64 * dt, &mut ctx)
                .expect("update");
        }

        // All written values must be finite.
        assert!(
            fields.iter().all(|v| v.is_finite()),
            "focused beam: field must remain finite"
        );

        // The on-axis centre point should have non-zero amplitude.
        let mid_y = grid.ny / 2;
        let mid_z = grid.nz / 2;
        let mid_x = grid.nx / 2;
        let p_centre = fields[[0, mid_x, mid_y, mid_z]];
        assert!(
            p_centre.abs() > 0.0,
            "focused beam: on-axis pressure must be non-zero, got {p_centre}"
        );
    }
}
