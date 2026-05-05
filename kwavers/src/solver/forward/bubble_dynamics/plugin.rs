//! Plugin adapter for bubble dynamics.
//!
//! Bridges the [`Plugin`] contract to the three production bubble-equation
//! implementations:
//!
//! | [`BubbleModel`] variant | ODE / integrator used |
//! |-------------------------|-----------------------|
//! | `KellerMiksis`          | [`BubbleField`] (adaptive KM, `use_compressibility = true`) |
//! | `RayleighPlesset`       | [`BubbleField`] (adaptive KM, `use_compressibility = false` → O(Mach⁰) limit) |
//! | `Gilmore`               | [`GilmoreSolver`] with per-voxel classical RK4, Tait liquid EOS |
//!
//! ## Field contract
//!
//! | Direction | [`UnifiedFieldType`] | Physical meaning |
//! |-----------|----------------------|-----------------|
//! | reads     | `Pressure`           | far-field acoustic driving pressure [Pa] |
//! | writes    | `BubbleRadius`       | instantaneous bubble radius R(t) [m] |
//! | writes    | `BubbleVelocity`     | bubble-wall velocity Ṙ(t) [m/s] |
//!
//! ## Nucleation seeding
//!
//! When `nucleation = false` (default), a single bubble is seeded at the
//! grid centre.  When `nucleation = true`, eight additional bubbles are
//! seeded at ±¼-domain offsets from the centre, modelling a focal zone
//! nucleation cloud.  All bubbles share the same `BubbleParameters`.
//!
//! ## dp/dt computation
//!
//! The [`BubbleField`] update requires `dp_dt_field` (the time derivative of
//! acoustic pressure) for the Keller-Miksis radiation-damping term.  This
//! plugin stores the previous-step pressure and computes a first-order
//! backward-difference estimate:
//!
//! ```text
//! dp_dt[i,j,k] ≈ (p_n[i,j,k] − p_{n-1}[i,j,k]) / dt
//! ```
//!
//! On the first call the denominator is finite but the numerator is zero
//! (previous pressure initialised to the current pressure at `initialize`
//! time), giving `dp_dt = 0` for the first step — the correct cold-start
//! behaviour.
//!
//! ## References
//!
//! - Keller & Miksis (1980) J. Acoust. Soc. Am. 68(2):628–633.
//! - Gilmore (1952) Caltech Hydrodynamics Lab Report 26-4.
//! - Rayleigh (1917) Phil. Mag. 34:94.

use std::collections::HashMap;

use ndarray::{Array3, Axis};

use crate::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use crate::physics::acoustics::bubble_dynamics::{
    bubble_field::BubbleField,
    bubble_state::{BubbleParameters, BubbleState},
    gilmore::GilmoreSolver,
};
use crate::physics::factory::models::BubbleModel;
use ndarray::Array4;

// ── Configuration ────────────────────────────────────────────────────────────

/// Construction-time configuration for [`BubbleDynamicsPlugin`].
#[derive(Debug, Clone)]
pub struct BubbleDynamicsConfig {
    /// Which ODE model to use for the bubble-wall motion.
    pub model: BubbleModel,
    /// When `true`, seed eight additional bubbles at ±¼-domain offsets from
    /// the grid centre (focal-zone nucleation cloud).
    pub nucleation: bool,
    /// Physical parameters shared by all seeded bubbles.
    pub params: BubbleParameters,
}

impl Default for BubbleDynamicsConfig {
    fn default() -> Self {
        Self {
            model: BubbleModel::KellerMiksis,
            nucleation: false,
            params: BubbleParameters::default(),
        }
    }
}

// ── Engine enum ───────────────────────────────────────────────────────────────

/// Per-model runtime engine, created lazily in [`Plugin::initialize`].
///
/// `KmOrRp` covers both Keller-Miksis and Rayleigh-Plesset because the
/// existing [`BubbleField`] code path handles both: when
/// `BubbleParameters::use_compressibility = false` the KM O(Mach) correction
/// factors collapse to unity, recovering the incompressible RP equation.
///
/// `Gilmore` drives per-voxel integration via [`GilmoreSolver::step_rk4`] —
/// the RK4 loop lives inside the solver where it belongs (SRP), not here.
/// The Gilmore path does **not** store a `prev_pressure` field because the
/// Gilmore ODE receives the instantaneous pressure at each voxel directly from
/// the field array; no dp/dt estimate is required.
enum BubbleEngine {
    /// Keller-Miksis or Rayleigh-Plesset via existing adaptive BubbleField.
    KmOrRp {
        field: BubbleField,
        /// Previous-step pressure, used to estimate dp/dt via backward difference.
        ///
        /// The [`BubbleField::update`] signature requires `dp_dt_field` for the
        /// KM radiation-damping term.  A first-order backward difference
        /// `dp_dt[i,j,k] ≈ (p_n − p_{n-1}) / dt` is sufficient for the
        /// O(Mach) accuracy level of the KM equation.
        prev_pressure: Array3<f64>,
    },
    /// Gilmore equation (Tait EOS) via per-voxel [`GilmoreSolver::step_rk4`].
    ///
    /// State carries only the live solver and the per-voxel `BubbleState`
    /// map. `BubbleParameters` are owned by the surrounding `BubblePluginConfig`
    /// (`self.config.params`) and read from there when needed; carrying a copy
    /// in this variant would duplicate the SSOT held by the plugin.
    Gilmore {
        solver: GilmoreSolver,
        /// Per-voxel bubble states, keyed by grid index (i, j, k).
        states: HashMap<(usize, usize, usize), BubbleState>,
    },
}

impl std::fmt::Debug for BubbleEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BubbleEngine::KmOrRp { .. } => write!(f, "BubbleEngine::KmOrRp"),
            BubbleEngine::Gilmore { .. } => write!(f, "BubbleEngine::Gilmore"),
        }
    }
}

// ── Plugin ────────────────────────────────────────────────────────────────────

/// Plugin adapter for bubble dynamics.
///
/// See module-level documentation for the field contract, nucleation model,
/// and dp/dt computation.
#[derive(Debug)]
pub struct BubbleDynamicsPlugin {
    metadata: PluginMetadata,
    config: BubbleDynamicsConfig,
    engine: Option<BubbleEngine>,
    state: PluginState,
}

impl BubbleDynamicsPlugin {
    /// Construct from a catalog-level config.
    ///
    /// The engine is `None` until [`Plugin::initialize`] is called; grid
    /// geometry is not yet available at construction time.
    #[must_use]
    pub fn new(config: BubbleDynamicsConfig) -> Self {
        let model_name = match &config.model {
            BubbleModel::KellerMiksis => "KellerMiksis",
            BubbleModel::RayleighPlesset => "RayleighPlesset",
            BubbleModel::Gilmore => "Gilmore",
        };
        Self {
            metadata: PluginMetadata {
                id: format!("bubble_dynamics_{}", model_name.to_lowercase()),
                name: format!("BubbleDynamics[{model_name}]"),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: format!(
                    "Bubble dynamics plugin ({model_name} ODE). \
                     Reads Pressure; writes BubbleRadius and BubbleVelocity."
                ),
                license: "MIT".to_string(),
            },
            config,
            engine: None,
            state: PluginState::Created,
        }
    }

    /// Seed positions for the initial bubble cloud.
    ///
    /// `nucleation = false` → 1 bubble at grid centre.
    /// `nucleation = true`  → 9 bubbles: centre + 8 corners at ±¼-domain.
    fn seed_positions(grid: &Grid, nucleation: bool) -> Vec<(usize, usize, usize)> {
        let cx = grid.nx / 2;
        let cy = grid.ny / 2;
        let cz = grid.nz / 2;
        let mut positions = vec![(cx, cy, cz)];
        if nucleation {
            let qx = (grid.nx / 4).max(1);
            let qy = (grid.ny / 4).max(1);
            let qz = (grid.nz / 4).max(1);
            for &ix in &[cx.saturating_sub(qx), (cx + qx).min(grid.nx - 1)] {
                for &iy in &[cy.saturating_sub(qy), (cy + qy).min(grid.ny - 1)] {
                    for &iz in &[cz.saturating_sub(qz), (cz + qz).min(grid.nz - 1)] {
                        let pos = (ix, iy, iz);
                        if pos != (cx, cy, cz) {
                            positions.push(pos);
                        }
                    }
                }
            }
        }
        positions
    }

    /// Write current bubble states back into the unified field array.
    ///
    /// Bounds-checks axis-0 size before every write so a field array that
    /// does not allocate the bubble channels is handled gracefully rather
    /// than panicking.
    fn write_bubble_fields(
        fields: &mut Array4<f64>,
        states: impl Iterator<Item = ((usize, usize, usize), (f64, f64))>,
    ) {
        let radius_idx = UnifiedFieldType::BubbleRadius.index();
        let vel_idx = UnifiedFieldType::BubbleVelocity.index();
        let n_fields = fields.shape()[0];

        for ((i, j, k), (radius, velocity)) in states {
            if radius_idx < n_fields {
                fields[[radius_idx, i, j, k]] = radius;
            }
            if vel_idx < n_fields {
                fields[[vel_idx, i, j, k]] = velocity;
            }
        }
    }
}

impl Plugin for BubbleDynamicsPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::BubbleRadius,
            UnifiedFieldType::BubbleVelocity,
        ]
    }

    fn initialize(&mut self, grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        let shape = (grid.nx, grid.ny, grid.nz);
        let prev_pressure = Array3::zeros(shape);
        let positions = Self::seed_positions(grid, self.config.nucleation);

        self.engine = Some(match &self.config.model {
            BubbleModel::KellerMiksis | BubbleModel::RayleighPlesset => {
                let mut params = self.config.params.clone();
                if matches!(self.config.model, BubbleModel::RayleighPlesset) {
                    // Disable compressibility correction → RP limit of KM equation.
                    params.use_compressibility = false;
                }
                let mut field =
                    BubbleField::with_spacing(shape, params.clone(), (grid.dx, grid.dy, grid.dz));
                for (i, j, k) in &positions {
                    field.add_bubble(*i, *j, *k, BubbleState::new(&params));
                }
                BubbleEngine::KmOrRp {
                    field,
                    prev_pressure,
                }
            }
            BubbleModel::Gilmore => {
                let solver = GilmoreSolver::new(self.config.params.clone());
                let states: HashMap<_, _> = positions
                    .iter()
                    .map(|&pos| (pos, BubbleState::new(&self.config.params)))
                    .collect();
                // prev_pressure is not needed for the Gilmore path: the
                // instantaneous acoustic field value is read directly at each
                // voxel position; no dp/dt estimate is required.
                drop(prev_pressure);
                BubbleEngine::Gilmore { solver, states }
            }
        });

        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let n_fields = fields.shape()[0];
        if pressure_idx >= n_fields {
            // Pressure channel absent — nothing to drive the bubbles with.
            return Ok(());
        }

        // Extract current pressure as an owned Array3.
        let current_pressure = fields.index_axis(Axis(0), pressure_idx).to_owned();

        match self.engine.as_mut() {
            None => {} // initialize() not yet called; skip silently.

            Some(BubbleEngine::KmOrRp {
                field,
                prev_pressure,
            }) => {
                // Backward-difference dp/dt estimate.
                let dp_dt = (&current_pressure - &*prev_pressure) / dt;

                field.update(&current_pressure, &dp_dt, dt, t);

                // Write bubble radius and wall velocity back to the field array.
                let state_iter = field
                    .bubbles
                    .iter()
                    .map(|(&pos, s)| (pos, (s.radius, s.wall_velocity)));
                Self::write_bubble_fields(fields, state_iter);

                *prev_pressure = current_pressure;
            }

            Some(BubbleEngine::Gilmore { solver, states }) => {
                // Per-voxel classical RK4 via GilmoreSolver::step_rk4.
                // The instantaneous acoustic field value at each voxel is passed
                // directly; dp/dt is not required by the Gilmore ODE.
                let updated: Vec<_> = states
                    .iter()
                    .map(|(&pos, state)| {
                        let (i, j, k) = pos;
                        let p_acoustic = current_pressure[[i, j, k]];
                        let next = solver.step_rk4(state, p_acoustic, t, dt);
                        (pos, next)
                    })
                    .collect();

                for (pos, next) in updated {
                    states.insert(pos, next);
                }

                let state_iter = states
                    .iter()
                    .map(|(&pos, s)| (pos, (s.radius, s.wall_velocity)));
                Self::write_bubble_fields(fields, state_iter);
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;
    use crate::domain::plugin::test_support::{make_context, null_plugin_fields, NullBoundary};
    use crate::physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
    use ndarray::Array4;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn small_grid() -> Grid {
        Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid")
    }

    fn water(grid: &Grid) -> HomogeneousMedium {
        HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, grid)
    }

    /// Allocate a field array with all required channels initialised to
    /// physiologically plausible values.
    ///
    /// Channels: [Pressure(0), Temperature(1), BubbleRadius(2), BubbleVelocity(3),
    ///            Density(4), SoundSpeed(5)]
    fn field_array(grid: &Grid) -> Array4<f64> {
        let n_fields = 6;
        let mut f = Array4::zeros((n_fields, grid.nx, grid.ny, grid.nz));
        // Seed Pressure channel with a modest 50 kPa driving pressure.
        f.index_axis_mut(Axis(0), UnifiedFieldType::Pressure.index())
            .fill(50_000.0);
        // Initialise BubbleRadius to equilibrium (5 µm default).
        f.index_axis_mut(Axis(0), UnifiedFieldType::BubbleRadius.index())
            .fill(5e-6);
        f
    }

    // ── KellerMiksis ─────────────────────────────────────────────────────────

    #[test]
    fn km_plugin_initialises_and_registers_correct_fields() {
        let config = BubbleDynamicsConfig {
            model: BubbleModel::KellerMiksis,
            nucleation: false,
            params: BubbleParameters::default(),
        };
        let mut plugin = BubbleDynamicsPlugin::new(config);
        let grid = small_grid();
        let medium = water(&grid);

        assert_eq!(plugin.required_fields(), vec![UnifiedFieldType::Pressure]);
        assert_eq!(
            plugin.provided_fields(),
            vec![
                UnifiedFieldType::BubbleRadius,
                UnifiedFieldType::BubbleVelocity
            ]
        );

        plugin
            .initialize(&grid, &medium)
            .expect("KM init must succeed");
        assert_eq!(plugin.state(), PluginState::Initialized);
    }

    #[test]
    fn km_plugin_writes_nonzero_radius_after_update() {
        let config = BubbleDynamicsConfig {
            model: BubbleModel::KellerMiksis,
            nucleation: false,
            params: BubbleParameters::default(),
        };
        let mut plugin = BubbleDynamicsPlugin::new(config);
        let grid = small_grid();
        let medium = water(&grid);
        plugin.initialize(&grid, &medium).expect("init");

        let mut fields = field_array(&grid);
        let extra_fields = null_plugin_fields(&grid);
        let mut null_boundary = NullBoundary;
        let mut ctx = make_context(&extra_fields, &mut null_boundary);
        plugin
            .update(&mut fields, &grid, &medium, 1e-7, 0.0, &mut ctx)
            .expect("KM update must succeed");

        // Centre voxel bubble radius must be a positive real value.
        let cx = grid.nx / 2;
        let cy = grid.ny / 2;
        let cz = grid.nz / 2;
        let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
        assert!(
            r > 0.0,
            "BubbleRadius at centre must be positive after KM update; got {r}"
        );
    }

    // ── RayleighPlesset ───────────────────────────────────────────────────────

    #[test]
    fn rp_plugin_initialises_and_advances() {
        let mut params = BubbleParameters::default();
        params.driving_amplitude = 30_000.0; // 30 kPa — well within RP regime
        let config = BubbleDynamicsConfig {
            model: BubbleModel::RayleighPlesset,
            nucleation: false,
            params,
        };
        let mut plugin = BubbleDynamicsPlugin::new(config);
        let grid = small_grid();
        let medium = water(&grid);
        plugin.initialize(&grid, &medium).expect("RP init");

        let mut fields = field_array(&grid);
        let extra_fields = null_plugin_fields(&grid);
        let mut null_boundary = NullBoundary;
        let mut ctx = make_context(&extra_fields, &mut null_boundary);
        plugin
            .update(&mut fields, &grid, &medium, 1e-7, 0.0, &mut ctx)
            .expect("RP update");

        let cx = grid.nx / 2;
        let cy = grid.ny / 2;
        let cz = grid.nz / 2;
        let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
        assert!(r > 0.0, "RP bubble radius must be positive; got {r}");
    }

    // ── Gilmore ───────────────────────────────────────────────────────────────

    #[test]
    fn gilmore_plugin_initialises_and_advances() {
        let config = BubbleDynamicsConfig {
            model: BubbleModel::Gilmore,
            nucleation: false,
            params: BubbleParameters::default(),
        };
        let mut plugin = BubbleDynamicsPlugin::new(config);
        let grid = small_grid();
        let medium = water(&grid);
        plugin.initialize(&grid, &medium).expect("Gilmore init");

        let mut fields = field_array(&grid);
        let extra_fields = null_plugin_fields(&grid);
        let mut null_boundary = NullBoundary;
        let mut ctx = make_context(&extra_fields, &mut null_boundary);
        plugin
            .update(&mut fields, &grid, &medium, 1e-7, 0.0, &mut ctx)
            .expect("Gilmore update");

        let cx = grid.nx / 2;
        let cy = grid.ny / 2;
        let cz = grid.nz / 2;
        let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
        assert!(r > 0.0, "Gilmore bubble radius must be positive; got {r}");
    }

    // ── Nucleation seeding ────────────────────────────────────────────────────

    #[test]
    fn nucleation_false_seeds_exactly_one_bubble() {
        let config = BubbleDynamicsConfig {
            model: BubbleModel::KellerMiksis,
            nucleation: false,
            params: BubbleParameters::default(),
        };
        let mut plugin = BubbleDynamicsPlugin::new(config);
        let grid = small_grid();
        let medium = water(&grid);
        plugin.initialize(&grid, &medium).expect("init");

        if let Some(BubbleEngine::KmOrRp { field, .. }) = &plugin.engine {
            assert_eq!(
                field.bubbles.len(),
                1,
                "nucleation=false must seed exactly 1 bubble; got {}",
                field.bubbles.len()
            );
        } else {
            panic!("expected KmOrRp engine");
        }
    }

    #[test]
    fn nucleation_true_seeds_multiple_bubbles() {
        let config = BubbleDynamicsConfig {
            model: BubbleModel::KellerMiksis,
            nucleation: true,
            params: BubbleParameters::default(),
        };
        let mut plugin = BubbleDynamicsPlugin::new(config);
        let grid = small_grid();
        let medium = water(&grid);
        plugin.initialize(&grid, &medium).expect("init");

        if let Some(BubbleEngine::KmOrRp { field, .. }) = &plugin.engine {
            assert!(
                field.bubbles.len() > 1,
                "nucleation=true must seed more than 1 bubble; got {}",
                field.bubbles.len()
            );
        } else {
            panic!("expected KmOrRp engine");
        }
    }

    // ── Multiple update steps ─────────────────────────────────────────────────

    #[test]
    fn km_plugin_radius_changes_over_three_steps() {
        let config = BubbleDynamicsConfig {
            model: BubbleModel::KellerMiksis,
            nucleation: false,
            params: BubbleParameters::default(),
        };
        let mut plugin = BubbleDynamicsPlugin::new(config);
        let grid = small_grid();
        let medium = water(&grid);
        plugin.initialize(&grid, &medium).expect("init");

        let dt = 1e-7;
        let mut fields = field_array(&grid);
        let extra_fields = null_plugin_fields(&grid);
        let mut null_boundary = NullBoundary;
        let cx = grid.nx / 2;
        let cy = grid.ny / 2;
        let cz = grid.nz / 2;

        let r0 = BubbleParameters::default().r0;
        let mut prev_r = r0;
        let mut any_changed = false;
        for step in 0..3 {
            let mut ctx = make_context(&extra_fields, &mut null_boundary);
            plugin
                .update(&mut fields, &grid, &medium, dt, step as f64 * dt, &mut ctx)
                .expect("update");
            let r = fields[[UnifiedFieldType::BubbleRadius.index(), cx, cy, cz]];
            assert!(r > 0.0, "radius must be positive at step {step}; got {r}");
            if (r - prev_r).abs() > 1e-15 {
                any_changed = true;
            }
            prev_r = r;
        }
        assert!(
            any_changed,
            "bubble radius must change over multiple steps under acoustic driving"
        );
    }
}
