//! Core Kuznetsov equation solver implementation
//!
//! Implements the full Kuznetsov equation for nonlinear acoustic wave propagation:
//! ∇²p - (1/c₀²)∂²p/∂t² = -(β/ρ₀c₀⁴)∂²p²/∂t² - (δ/c₀⁴)∂³p/∂t³ + F

use super::config::KuznetsovConfig;
use super::diffusion::compute_diffusive_term_workspace;
use super::nonlinear::compute_nonlinear_term_workspace;
use super::workspace::KuznetsovWorkspace;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::physics::traits::AcousticWaveModel;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker, ViolationSeverity,
};
use log;
use ndarray::{Array3, Array4, Zip};

/// Main Kuznetsov wave solver
#[derive(Debug)]
pub struct KuznetsovWave {
    config: KuznetsovConfig,
    grid: Grid,
    workspace: KuznetsovWorkspace,
    nonlinearity_scaling: f64,
    time_step_count: usize,
    /// Previous pressure field for leapfrog time integration
    pressure_prev: Array3<f64>,
    /// Current pressure field
    pressure_current: Array3<f64>,
    /// Flag to track if this is the first time step
    first_step: bool,
    /// Conservation diagnostics tracker
    conservation_tracker: Option<ConservationTracker>,
    /// Current simulation time
    current_time: f64,
    /// Cached medium properties for conservation calculations
    medium_properties: MediumProperties,
}

/// Cached medium properties for conservation calculations
#[derive(Debug, Clone)]
struct MediumProperties {
    rho0: f64,
    c0: f64,
}

impl KuznetsovWave {
    /// Create a new Kuznetsov wave solver
    pub fn new(config: KuznetsovConfig, grid: &Grid) -> KwaversResult<Self> {
        config.validate(grid)?;
        let workspace = KuznetsovWorkspace::new(grid)?;
        let shape = (grid.nx, grid.ny, grid.nz);

        // Get representative medium properties (default to water)
        let rho0 = 1000.0; // Default water density
        let c0 = 1500.0; // Default water sound speed

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

    /// Enable conservation diagnostics with specified tolerances
    ///
    /// # Arguments
    ///
    /// * `tolerances` - Conservation tolerance parameters (absolute/relative/check_interval)
    /// * `medium` - Medium for extracting representative properties
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kwavers::solver::forward::nonlinear::conservation::ConservationTolerances;
    ///
    /// let mut solver = KuznetsovWave::new(config, &grid)?;
    /// solver.enable_conservation_diagnostics(ConservationTolerances::default(), &medium);
    /// ```
    pub fn enable_conservation_diagnostics(
        &mut self,
        tolerances: ConservationTolerances,
        medium: &dyn Medium,
    ) {
        // Update medium properties from actual medium
        let center_x = self.grid.dx * (self.grid.nx as f64) / 2.0;
        let center_y = self.grid.dy * (self.grid.ny as f64) / 2.0;
        let center_z = self.grid.dz * (self.grid.nz as f64) / 2.0;
        self.medium_properties.rho0 =
            crate::domain::medium::density_at(medium, center_x, center_y, center_z, &self.grid);
        self.medium_properties.c0 =
            crate::domain::medium::sound_speed_at(medium, center_x, center_y, center_z, &self.grid);

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

    /// Disable conservation diagnostics
    pub fn disable_conservation_diagnostics(&mut self) {
        self.conservation_tracker = None;
    }

    /// Get conservation diagnostic summary
    ///
    /// Returns a summary of all conservation checks performed,
    /// including maximum severity and error magnitudes.
    pub fn get_conservation_summary(&self) -> Option<String> {
        self.conservation_tracker
            .as_ref()
            .map(|tracker| tracker.summary().to_string())
    }

    /// Check if solution satisfies conservation constraints
    ///
    /// Returns `true` if all conservation violations are within acceptable limits.
    pub fn is_solution_valid(&self) -> bool {
        self.conservation_tracker
            .as_ref()
            .is_none_or(|tracker| tracker.is_solution_valid())
    }

    /// Compute the right-hand side of the Kuznetsov equation
    ///
    /// For heterogeneous media, nonlinear and diffusive terms are computed using
    /// local material properties at each grid point. For homogeneous media,
    /// properties are computed once for efficiency.
    fn compute_rhs(
        &mut self,
        pressure: &Array3<f64>,
        source: &dyn Source,
        medium: &dyn Medium,
        t: f64,
        dt: f64,
    ) -> Array3<f64> {
        // Pre-compute medium properties for all grid points if heterogeneous
        let is_heterogeneous = !medium.is_homogeneous();

        // For homogeneous media, compute properties once
        let (uniform_density, uniform_sound_speed, _uniform_nonlinearity, uniform_diffusivity) =
            if !is_heterogeneous {
                let center_x = self.grid.dx * (self.grid.nx as f64) / 2.0;
                let center_y = self.grid.dy * (self.grid.ny as f64) / 2.0;
                let center_z = self.grid.dz * (self.grid.nz as f64) / 2.0;
                (
                    crate::domain::medium::density_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    crate::domain::medium::sound_speed_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    crate::domain::medium::nonlinearity_at(
                        medium, center_x, center_y, center_z, &self.grid,
                    ),
                    self.config.acoustic_diffusivity,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0) // Will compute per-point
            };

        // 1. Compute linear term: c₀²∇²p using spectral methods
        self.workspace.spectral_op.compute_laplacian_workspace(
            pressure,
            &mut self.workspace.laplacian,
            &self.grid,
        );

        let mut rhs = Array3::zeros(pressure.dim());

        // 2. Compute nonlinear term if enabled
        let include_nonlinearity = matches!(
            self.config.equation_mode,
            super::config::AcousticEquationMode::FullKuznetsov
                | super::config::AcousticEquationMode::KZK
                | super::config::AcousticEquationMode::Westervelt
        );

        if include_nonlinearity && !is_heterogeneous {
            // For homogeneous media, compute once
            compute_nonlinear_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                dt,
                uniform_density,
                uniform_sound_speed,
                self.config.nonlinearity_coefficient * self.nonlinearity_scaling,
                &mut self.workspace.nonlinear_term,
            );
        }

        // 3. Compute diffusive term if enabled
        let include_diffusion = matches!(
            self.config.equation_mode,
            super::config::AcousticEquationMode::FullKuznetsov
                | super::config::AcousticEquationMode::KZK
        ) && self.config.acoustic_diffusivity > 0.0;

        if include_diffusion && !is_heterogeneous {
            // For homogeneous media, compute once
            compute_diffusive_term_workspace(
                pressure,
                &self.workspace.pressure_prev,
                &self.workspace.pressure_prev2,
                &self.workspace.pressure_prev3,
                dt,
                uniform_sound_speed,
                uniform_diffusivity,
                &mut self.workspace.diffusive_term,
            );
        }

        // 4. Combine all terms using efficient iteration
        if is_heterogeneous {
            // HETEROGENEOUS MEDIA: Compute properties per-point

            // Use parallel iteration with indices
            let shape = rhs.shape();
            let _nx = shape[0];
            let _ny = shape[1];
            let _nz = shape[2];

            // Use parallel iteration for heterogeneous media
            for k in 0..self.grid.nz {
                for j in 0..self.grid.ny {
                    for i in 0..self.grid.nx {
                        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);

                        // Get local medium properties
                        let local_density =
                            crate::domain::medium::density_at(medium, x, y, z, &self.grid);
                        let local_sound_speed =
                            crate::domain::medium::sound_speed_at(medium, x, y, z, &self.grid);
                        let local_nonlinearity =
                            crate::domain::medium::nonlinearity_at(medium, x, y, z, &self.grid);
                        let c0_squared = local_sound_speed * local_sound_speed;

                        // Linear term with local sound speed
                        rhs[[i, j, k]] = c0_squared * self.workspace.laplacian[[i, j, k]];

                        // Add nonlinear term with local properties
                        if include_nonlinearity {
                            // Full Kuznetsov nonlinear term: -(β/ρ₀c₀⁴) ∂²p²/∂t²
                            let beta = crate::physics::constants::NONLINEARITY_COEFFICIENT_OFFSET
                                + local_nonlinearity / crate::physics::constants::B_OVER_A_DIVISOR;
                            let coeff = beta / (local_density * local_sound_speed.powi(4));

                            // Compute p² at each time step
                            let p2 = pressure[[i, j, k]] * pressure[[i, j, k]];
                            let p2_prev = self.workspace.pressure_prev[[i, j, k]]
                                * self.workspace.pressure_prev[[i, j, k]];
                            let p2_prev2 = self.workspace.pressure_prev2[[i, j, k]]
                                * self.workspace.pressure_prev2[[i, j, k]];

                            // Second time derivative of p²
                            let d2p2_dt2 = (p2 - 2.0 * p2_prev + p2_prev2) / (dt * dt);

                            let nonlinear = -coeff * d2p2_dt2;
                            rhs[[i, j, k]] += nonlinear; // No clamping - allow natural shock formation
                        }

                        // Add diffusive term with local properties
                        if include_diffusion {
                            // Simplified diffusion: δ ∂³p/∂t³
                            let d3p_dt3 = (pressure[[i, j, k]]
                                - 3.0 * self.workspace.pressure_prev[[i, j, k]]
                                + 3.0 * self.workspace.pressure_prev2[[i, j, k]]
                                - self.workspace.pressure_prev3[[i, j, k]])
                                / dt.powi(3);
                            rhs[[i, j, k]] += self.config.acoustic_diffusivity * d3p_dt3;
                        }

                        // Add source term
                        rhs[[i, j, k]] += source.get_source_term(t, x, y, z, &self.grid);
                    }
                }
            }
        } else {
            // HOMOGENEOUS MEDIA: Use pre-computed uniform properties
            let c0_squared = uniform_sound_speed * uniform_sound_speed;

            // Use ndarray::Zip for efficient iteration

            Zip::from(&mut rhs)
                .and(&self.workspace.laplacian)
                .par_for_each(|r, &lap| {
                    // Linear term
                    *r = c0_squared * lap;
                });

            // Add source term separately to avoid index complications
            for k in 0..self.grid.nz {
                for j in 0..self.grid.ny {
                    for i in 0..self.grid.nx {
                        let (x, y, z) = self.grid.indices_to_coordinates(i, j, k);
                        rhs[[i, j, k]] += source.get_source_term(t, x, y, z, &self.grid);
                    }
                }
            }

            // Add nonlinear term if computed
            if include_nonlinearity {
                Zip::from(&mut rhs)
                    .and(&self.workspace.nonlinear_term)
                    .par_for_each(|r, &nl| *r += nl);
            }

            // Add diffusive term if computed
            if include_diffusion {
                Zip::from(&mut rhs)
                    .and(&self.workspace.diffusive_term)
                    .par_for_each(|r, &diff| *r += diff);
            }
        }

        rhs
    }
}

impl AcousticWaveModel for KuznetsovWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>, // Ignored - we manage our own state
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        // Validate grid consistency
        if grid.nx != self.grid.nx || grid.ny != self.grid.ny || grid.nz != self.grid.nz {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions mismatch with solver initialization".to_string(),
            ));
        }

        // Add warning for heterogeneous media
        if !medium.is_homogeneous() {
            log::warn!("Kuznetsov solver: Heterogeneous media support is incomplete. Nonlinear and diffusive terms use averaged properties, leading to inaccurate results.");
        }

        // Extract pressure field (assuming it's at index 0)
        let mut pressure_field = fields.index_axis_mut(ndarray::Axis(0), 0);

        if self.first_step {
            // First step: use forward Euler to bootstrap the leapfrog scheme
            self.pressure_current.assign(&pressure_field);
            self.pressure_prev.assign(&pressure_field); // Initialize both to current

            // Compute RHS
            let pressure_current = self.pressure_current.clone();
            let rhs = self.compute_rhs(&pressure_current, source, medium, t, dt);

            // Forward Euler for first step: p_next = p_current + 0.5 * dt^2 * acceleration
            // (assuming zero initial velocity)
            Zip::from(&mut pressure_field)
                .and(&self.pressure_current)
                .and(&rhs)
                .par_for_each(|p_next, &p_curr, &accel| {
                    *p_next = p_curr + 0.5 * dt * dt * accel;
                });

            // Update internal state
            self.workspace.update_time_history(&self.pressure_current);
            self.first_step = false;
        } else {
            // Leapfrog time integration for second-order wave equation
            // p_next = 2 * p_current - p_previous + dt^2 * rhs

            // Compute RHS using current pressure
            let pressure_current = self.pressure_current.clone();
            let rhs = self.compute_rhs(&pressure_current, source, medium, t, dt);

            // Apply leapfrog scheme using internal state
            Zip::from(&mut pressure_field)
                .and(&self.pressure_current)
                .and(&self.pressure_prev)
                .and(&rhs)
                .par_for_each(|p_next, &p_curr, &p_prev, &accel| {
                    *p_next = 2.0 * p_curr - p_prev + dt * dt * accel;
                });

            // Update internal state for next iteration
            self.pressure_prev.assign(&self.pressure_current);
            self.pressure_current.assign(&pressure_field);
            self.workspace.update_time_history(&self.pressure_current);
        }

        self.time_step_count += 1;
        self.current_time += dt;

        // Conservation diagnostics (if enabled)
        self.check_conservation_laws();

        Ok(())
    }

    fn report_performance(&self) {
        println!("KuznetsovWave solver performance:");
        println!("  Configuration: {:?}", self.config.equation_mode);
        println!(
            "  Grid size: {}x{}x{}",
            self.grid.nx, self.grid.ny, self.grid.nz
        );
        println!("  Time steps completed: {}", self.time_step_count);
        println!(
            "  Nonlinearity coefficient: {}",
            self.config.nonlinearity_coefficient
        );
        println!(
            "  Acoustic diffusivity: {:.2e}",
            self.config.acoustic_diffusivity
        );
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.nonlinearity_scaling = scaling;
    }
}

/// Check conservation laws and log diagnostics
///
/// Performs conservation checks at configured intervals and logs violations.
/// Critical violations trigger warnings via tracing infrastructure.
impl KuznetsovWave {
    fn check_conservation_laws(&mut self) {
        // Check if we should perform diagnostics at this step
        let should_check = self.conservation_tracker.as_ref().is_some_and(|tracker| {
            self.time_step_count
                .is_multiple_of(tracker.tolerances.check_interval)
        });

        if !should_check {
            return;
        }

        // Compute diagnostics first (without holding mutable reference to tracker)
        let (initial_energy, initial_momentum, initial_mass, tolerances) =
            if let Some(ref tracker) = self.conservation_tracker {
                (
                    tracker.initial_energy,
                    tracker.initial_momentum,
                    tracker.initial_mass,
                    tracker.tolerances,
                )
            } else {
                return;
            };

        let diagnostics = self.check_all_conservation(
            initial_energy,
            initial_momentum,
            initial_mass,
            self.time_step_count,
            self.current_time,
            &tolerances,
        );

        // Now update tracker with diagnostics
        if let Some(ref mut tracker) = self.conservation_tracker {
            // Update max severity
            for diag in &diagnostics {
                if diag.severity > tracker.max_severity {
                    tracker.max_severity = diag.severity;
                }
            }
            // Store in history
            tracker.history.extend(diagnostics.clone());
        }

        // Log diagnostics based on severity
        for diag in diagnostics {
            match diag.severity {
                ViolationSeverity::Acceptable => {
                    // Silent for acceptable violations
                }
                ViolationSeverity::Warning => {
                    eprintln!("⚠️  Kuznetsov Conservation Warning: {}", diag);
                }
                ViolationSeverity::Error => {
                    eprintln!("❌ Kuznetsov Conservation Error: {}", diag);
                }
                ViolationSeverity::Critical => {
                    eprintln!("🔴 Kuznetsov Conservation CRITICAL: {}", diag);
                    eprintln!("   Solution may be physically invalid!");
                }
            }
        }
    }
}

/// Implementation of conservation diagnostics trait for Kuznetsov solver
impl ConservationDiagnostics for KuznetsovWave {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²)
        // Total energy: ∫∫∫ E dV
        let mut total_energy = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let factor = 1.0 / (2.0 * rho0 * c0 * c0);
        let dv = self.grid.dx * self.grid.dy * self.grid.dz; // Volume element

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure_current[[i, j, k]];
                    total_energy += p * p * factor * dv;
                }
            }
        }

        total_energy
    }

    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        // Full 3D momentum calculation
        // Momentum density: ρ₀ u where u = ∫ ∇p/(ρ₀) dt (acoustic approximation)
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;

        // Compute momentum from pressure gradients
        for i in 1..self.grid.nx - 1 {
            for j in 1..self.grid.ny - 1 {
                for k in 1..self.grid.nz - 1 {
                    // Pressure gradients (central difference)
                    let dp_dx = (self.pressure_current[[i + 1, j, k]]
                        - self.pressure_current[[i - 1, j, k]])
                        / (2.0 * self.grid.dx);
                    let dp_dy = (self.pressure_current[[i, j + 1, k]]
                        - self.pressure_current[[i, j - 1, k]])
                        / (2.0 * self.grid.dy);
                    let dp_dz = (self.pressure_current[[i, j, k + 1]]
                        - self.pressure_current[[i, j, k - 1]])
                        / (2.0 * self.grid.dz);

                    // Momentum from pressure (acoustic approximation)
                    px += (rho0 * dp_dx / c0) * dv;
                    py += (rho0 * dp_dy / c0) * dv;
                    pz += (rho0 * dp_dz / c0) * dv;
                }
            }
        }

        (px, py, pz)
    }

    fn calculate_total_mass(&self) -> f64 {
        // For acoustic waves: ρ = ρ₀(1 + p/(ρ₀c₀²))
        // Total mass: ∫∫∫ ρ dV
        let mut total_mass = 0.0;
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    let p = self.pressure_current[[i, j, k]];
                    let rho = rho0 * (1.0 + p / (rho0 * c0 * c0));
                    total_mass += rho * dv;
                }
            }
        }

        total_mass
    }
}
