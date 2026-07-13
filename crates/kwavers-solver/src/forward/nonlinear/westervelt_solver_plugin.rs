//! `Plugin` adapter for the spectral Westervelt nonlinear solver.
//!
//! Wraps the validated [`WesterveltWave`] full-wave solver (Chapter 3,
//! `forward::nonlinear::westervelt_spectral`) behind the
//! [`Plugin`] contract so the `PhysicsCatalog` can build
//! it for `NonlinearAcoustics { equation_type: Westervelt }`. The adapter holds
//! no physics: it forwards `Plugin::update` to the solver's
//! [`AcousticWaveModel::update_wave`], exactly as `KzkSolverPlugin` does for KZK.

use crate::forward::nonlinear::westervelt_spectral::WesterveltWave;
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::traits::AcousticWaveModel;
use kwavers_source::NullSource;
use leto::{Array3, Array4};

/// Catalog plugin wrapping the spectral Westervelt solver.
#[derive(Debug)]
pub struct WesterveltSolverPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// Constructed in [`Plugin::initialize`] once the grid is known. The solver
    /// owns its second-order pressure history, so the plugin is otherwise
    /// stateless across steps.
    solver: Option<WesterveltWave>,
}

impl Default for WesterveltSolverPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl WesterveltSolverPlugin {
    /// Create a new (uninitialized) Westervelt plugin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "westervelt_solver".to_owned(),
                name: "Westervelt Equation Solver".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Nonlinear full-wave propagation via the spectral Westervelt equation"
                    .to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            solver: None,
        }
    }
}

impl Plugin for WesterveltSolverPlugin {
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

    fn initialize(&mut self, grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.solver = Some(WesterveltWave::new(grid));
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let solver = self.solver.as_mut().ok_or_else(|| {
            KwaversError::InvalidInput(
                "WesterveltSolverPlugin::update called before initialize".to_owned(),
            )
        })?;
        // The spectral Westervelt solver maintains its own pressure history; the
        // `prev_pressure` argument of `update_wave` is unused, so a zero field of
        // grid shape satisfies the contract. Sources are injected elsewhere in the
        // orchestration, so this propagation step uses the explicit null source.
        let prev_pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let source = NullSource::new();
        solver.update_wave(fields, &prev_pressure, &source, grid, medium, dt, t)
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.solver = None;
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
