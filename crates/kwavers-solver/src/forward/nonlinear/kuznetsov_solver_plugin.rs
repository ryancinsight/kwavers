//! `Plugin` adapter for the full Kuznetsov nonlinear solver.
//!
//! Wraps the validated [`KuznetsovWave`] solver (Chapter 3,
//! `forward::nonlinear::kuznetsov`) behind the [`Plugin`](crate::plugin::Plugin)
//! contract so the `PhysicsCatalog` can build it for
//! `NonlinearAcoustics { equation_type: Kuznetsov }`. The adapter holds no
//! physics: it forwards `Plugin::update` to the solver's
//! [`AcousticWaveModel::update_wave`], exactly as `KzkSolverPlugin` does for KZK.

use crate::forward::nonlinear::kuznetsov::{KuznetsovConfig, KuznetsovWave};
use crate::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::traits::AcousticWaveModel;
use kwavers_source::NullSource;
use ndarray::{Array3, Array4};

/// Catalog plugin wrapping the full Kuznetsov solver.
#[derive(Debug)]
pub struct KuznetsovSolverPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    config: KuznetsovConfig,
    /// Constructed in [`Plugin::initialize`] once the grid is known. The solver
    /// owns its time history, so the plugin is otherwise stateless across steps.
    solver: Option<KuznetsovWave>,
}

impl Default for KuznetsovSolverPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl KuznetsovSolverPlugin {
    /// Create a new (uninitialized) Kuznetsov plugin with the default config
    /// (nonlinearity and acoustic-diffusivity terms enabled).
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(KuznetsovConfig::default())
    }

    /// Create a Kuznetsov plugin with an explicit solver configuration.
    #[must_use]
    pub fn with_config(config: KuznetsovConfig) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "kuznetsov_solver".to_owned(),
                name: "Kuznetsov Equation Solver".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description:
                    "Nonlinear full-wave propagation via the Kuznetsov equation (nonlinearity + \
                     acoustic diffusivity)"
                        .to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            config,
            solver: None,
        }
    }
}

impl Plugin for KuznetsovSolverPlugin {
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
        self.solver = Some(KuznetsovWave::new(self.config.clone(), grid)?);
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
                "KuznetsovSolverPlugin::update called before initialize".to_owned(),
            )
        })?;
        // The Kuznetsov solver maintains its own time history; the `prev_pressure`
        // argument is satisfied by a zero field of grid shape. Sources are injected
        // elsewhere in the orchestration, so this step uses the null source.
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
