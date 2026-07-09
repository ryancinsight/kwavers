//! `KuznetsovWave` struct definition and primary impl methods.

use crate::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker,
};
use crate::forward::nonlinear::kuznetsov::config::KuznetsovConfig;
use crate::forward::nonlinear::kuznetsov::workspace::KuznetsovWorkspace;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array3;

/// Cached medium properties for conservation calculations
#[derive(Debug, Clone)]
pub(super) struct MediumProperties {
    pub(super) rho0: f64,
    pub(super) c0: f64,
}

/// Main Kuznetsov wave solver
#[derive(Debug)]
pub struct KuznetsovWave {
    pub(super) config: KuznetsovConfig,
    pub(super) grid: Grid,
    pub(super) workspace: KuznetsovWorkspace,
    pub(super) nonlinearity_scaling: f64,
    pub(super) time_step_count: usize,
    /// Previous pressure field for leapfrog time integration
    pub(super) pressure_prev: Array3<f64>,
    /// Current pressure field
    pub(super) pressure_current: Array3<f64>,
    /// Flag to track if this is the first time step
    pub(super) first_step: bool,
    /// Conservation diagnostics tracker
    pub(super) conservation_tracker: Option<ConservationTracker>,
    /// Current simulation time
    pub(super) current_time: f64,
    /// Cached medium properties for conservation calculations
    pub(super) medium_properties: MediumProperties,
}

impl KuznetsovWave {
    /// Create a new Kuznetsov wave solver
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: KuznetsovConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate(grid)?;
        let workspace = KuznetsovWorkspace::new(grid)?;
        let shape = (grid.nx, grid.ny, grid.nz);

        // Default to water properties
        let rho0 = DENSITY_WATER_NOMINAL;
        let c0 = SOUND_SPEED_WATER_SIM;

        Ok(Self {
            config,
            grid: grid.clone(),
            workspace,
            nonlinearity_scaling: 1.0,
            time_step_count: 0,
            pressure_prev: Array3::zeros(shape),
            pressure_current: Array3::zeros(shape),
            first_step: true,
            conservation_tracker: None,
            current_time: 0.0,
            medium_properties: MediumProperties { rho0, c0 },
        })
    }

    /// Enable conservation diagnostics with specified tolerances.
    pub fn enable_conservation_diagnostics(
        &mut self,
        tolerances: ConservationTolerances,
        medium: &dyn Medium,
    ) {
        let center_x = self.grid.dx * (self.grid.nx as f64) / 2.0;
        let center_y = self.grid.dy * (self.grid.ny as f64) / 2.0;
        let center_z = self.grid.dz * (self.grid.nz as f64) / 2.0;
        self.medium_properties.rho0 =
            kwavers_medium::density_at(medium, center_x, center_y, center_z, &self.grid);
        self.medium_properties.c0 =
            kwavers_medium::sound_speed_at(medium, center_x, center_y, center_z, &self.grid);

        // ConservationDiagnostics trait is in scope — impl is in diagnostics_impl.rs
        let initial_energy = self.calculate_total_energy();
        let initial_momentum = self.calculate_total_momentum();
        let initial_mass = self.calculate_total_mass();

        self.conservation_tracker = Some(ConservationTracker::new(
            initial_energy,
            initial_momentum,
            initial_mass,
            tolerances,
        ));
    }

    /// Disable conservation diagnostics.
    pub fn disable_conservation_diagnostics(&mut self) {
        self.conservation_tracker = None;
    }

    /// Get conservation diagnostic summary.
    pub fn get_conservation_summary(&self) -> Option<String> {
        self.conservation_tracker
            .as_ref()
            .map(|tracker| tracker.summary().to_string())
    }

    /// Check if solution satisfies conservation constraints.
    pub fn is_solution_valid(&self) -> bool {
        self.conservation_tracker
            .as_ref()
            .is_none_or(|tracker| tracker.is_solution_valid())
    }
}
