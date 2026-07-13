use std::collections::HashMap;

use leto::{Array3, Array4};

use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::KwaversResult;
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::bubble_dynamics::{
    bubble_field::BubbleField, bubble_state::BubbleState, gilmore::GilmoreSolver,
};
use kwavers_physics::factory::models::BubbleModel;

use super::config::BubbleDynamicsConfig;
use super::engine::BubbleEngine;

/// Plugin adapter for bubble dynamics.
///
/// See module-level documentation for the field contract, nucleation model,
/// and dp/dt computation.
#[derive(Debug)]
pub struct BubbleDynamicsPlugin {
    metadata: PluginMetadata,
    config: BubbleDynamicsConfig,
    pub(super) engine: Option<BubbleEngine>,
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
            BubbleModel::KellerHerring => "KellerHerring",
            BubbleModel::RayleighPlesset => "RayleighPlesset",
            BubbleModel::Gilmore => "Gilmore",
        };
        Self {
            metadata: PluginMetadata {
                id: format!("bubble_dynamics_{}", model_name.to_lowercase()),
                name: format!("BubbleDynamics[{model_name}]"),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: format!(
                    "Bubble dynamics plugin ({model_name} ODE). \
                     Reads Pressure; writes BubbleRadius and BubbleVelocity."
                ),
                license: "MIT".to_owned(),
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
            BubbleModel::KellerMiksis => {
                let params = self.config.params.clone();
                let mut field = BubbleField::with_keller_miksis(
                    shape,
                    params.clone(),
                    (grid.dx, grid.dy, grid.dz),
                );
                for (i, j, k) in &positions {
                    field.add_bubble(*i, *j, *k, BubbleState::new(&params));
                }
                BubbleEngine::KmOrRp {
                    field: Box::new(field),
                    prev_pressure,
                }
            }
            BubbleModel::KellerHerring => {
                let params = self.config.params.clone();
                let mut field = BubbleField::with_keller_herring(
                    shape,
                    params.clone(),
                    (grid.dx, grid.dy, grid.dz),
                );
                for (i, j, k) in &positions {
                    field.add_bubble(*i, *j, *k, BubbleState::new(&params));
                }
                BubbleEngine::KmOrRp {
                    field: Box::new(field),
                    prev_pressure,
                }
            }
            BubbleModel::RayleighPlesset => {
                let mut params = self.config.params.clone();
                // Disable compressibility correction → RP limit of KM equation.
                params.use_compressibility = false;
                let mut field = BubbleField::with_rayleigh_plesset(
                    shape,
                    params.clone(),
                    (grid.dx, grid.dy, grid.dz),
                );
                for (i, j, k) in &positions {
                    field.add_bubble(*i, *j, *k, BubbleState::new(&params));
                }
                BubbleEngine::KmOrRp {
                    field: Box::new(field),
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
        let pressure_view = fields
            .index_axis::<3>(0, pressure_idx)
            .expect("invariant: pressure field axis index in range");
        let current_pressure = Array3::from_shape_vec(
            pressure_view.shape(),
            pressure_view.iter().copied().collect(),
        )
        .expect("invariant: axis view shape yields valid owned array");

        match self.engine.as_mut() {
            None => {} // initialize() not yet called; skip silently.

            Some(BubbleEngine::KmOrRp {
                field,
                prev_pressure,
            }) => {
                // Backward-difference dp/dt estimate.
                let dp_diff = &current_pressure - &*prev_pressure;
                let dp_dt = &dp_diff / dt;

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
