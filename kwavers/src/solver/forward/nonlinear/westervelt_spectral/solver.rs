//! Main Westervelt equation solver implementation

use crate::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::physics::traits::AcousticWaveModel;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker, ViolationSeverity,
};
use ndarray::{Array3, Array4, Axis};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::metrics::PerformanceMetrics;
use super::nonlinear::{compute_nonlinear_term, compute_viscoelastic_term};
use super::spectral::{compute_laplacian_spectral, initialize_kspace_grids};

/// Westervelt equation solver with proper second-order time derivatives
#[derive(Debug)]
pub struct WesterveltWave {
    // Precomputed k-space grids
    k_squared: Option<Array3<f64>>,
    // Note: Individual k-space components removed as they were unused
    // k_squared contains kx^2 + ky^2 + kz^2 which is sufficient for Laplacian

    // Configuration
    nonlinearity_scaling: f64,
    // No explicit pressure clamping - allows natural shock formation through nonlinearity
    // Stability maintained through CFL conditions and physical viscoelastic damping

    // Pressure history using buffer rotation for zero-allocation updates
    pressure_buffers: [Array3<f64>; 3],
    buffer_indices: [usize; 3], // [next, current, previous]

    // Performance tracking
    metrics: Arc<Mutex<PerformanceMetrics>>,

    // Conservation diagnostics
    conservation_tracker: Option<ConservationTracker>,
    current_step: usize,
    current_time: f64,
    grid_cache: Option<Grid>,
    medium_properties: Option<MediumProperties>,
}

/// Cached medium properties for conservation calculations
#[derive(Debug, Clone)]
struct MediumProperties {
    rho0: f64,
    c0: f64,
}

impl WesterveltWave {
    /// Create a new Westervelt solver
    pub fn new(grid: &Grid) -> Self {
        let (k_squared, _kx, _ky, _kz) = initialize_kspace_grids(grid);
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            k_squared: Some(k_squared),
            nonlinearity_scaling: 1.0,
            pressure_buffers: [
                Array3::zeros(shape),
                Array3::zeros(shape),
                Array3::zeros(shape),
            ],
            buffer_indices: [0, 1, 2], // Initially: next=0, current=1, previous=2
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
            conservation_tracker: None,
            current_step: 0,
            current_time: 0.0,
            grid_cache: Some(grid.clone()),
            medium_properties: None,
        }
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
    /// let mut solver = WesterveltWave::new(&grid);
    /// solver.enable_conservation_diagnostics(ConservationTolerances::default(), &medium);
    /// ```
    pub fn enable_conservation_diagnostics(
        &mut self,
        tolerances: ConservationTolerances,
        medium: &dyn Medium,
    ) {
        // Cache medium properties
        if let Some(ref grid) = self.grid_cache {
            let center_x = grid.dx * (grid.nx as f64) / 2.0;
            let center_y = grid.dy * (grid.ny as f64) / 2.0;
            let center_z = grid.dz * (grid.nz as f64) / 2.0;
            let rho0 =
                crate::domain::medium::density_at(medium, center_x, center_y, center_z, grid);
            let c0 =
                crate::domain::medium::sound_speed_at(medium, center_x, center_y, center_z, grid);
            self.medium_properties = Some(MediumProperties { rho0, c0 });
        }

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

    /// Check stability of the current configuration
    fn check_stability(
        &self,
        dt: f64,
        grid: &Grid,
        medium: &dyn Medium,
        _pressure: &Array3<f64>,
    ) -> bool {
        // CFL condition ensures numerical stability for shock-capable nonlinear propagation
        let max_c = medium
            .sound_speed_array()
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x));
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_c * dt / min_dx;

        cfl < 0.5
    }

    /// Initialize pressure buffers from the field
    fn initialize_buffers(&mut self, initial_pressure: &Array3<f64>) {
        self.pressure_buffers[self.buffer_indices[1]].assign(initial_pressure);
        self.pressure_buffers[self.buffer_indices[2]].assign(initial_pressure);
    }

    /// Get performance summary
    #[must_use]
    pub fn performance_summary(&self) -> String {
        let metrics = self.metrics.lock().unwrap();
        format!(
            "WesterveltWave Performance: {} calls, {:.2} ms total",
            metrics.call_count,
            metrics.total_time() * 1000.0
        )
    }
}

impl AcousticWaveModel for WesterveltWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.increment_calls();
        }

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if nx == 0 || ny == 0 || nz == 0 {
            return Ok(());
        }

        // Initialize buffers on first call
        if self.buffer_indices[1] == self.buffer_indices[2] {
            let initial_pressure = fields.index_axis(Axis(0), UnifiedFieldType::Pressure.index());
            self.initialize_buffers(&initial_pressure.to_owned());
        }

        // Get buffer references based on current indices
        let (next_idx, curr_idx, prev_idx) = (
            self.buffer_indices[0],
            self.buffer_indices[1],
            self.buffer_indices[2],
        );

        // Get current pressure from the field
        let pressure_field = fields.index_axis(Axis(0), UnifiedFieldType::Pressure.index());
        self.pressure_buffers[curr_idx].assign(&pressure_field);

        // Clone the pressure data we need to avoid borrow conflicts
        let pressure_current = self.pressure_buffers[curr_idx].clone();
        let pressure_previous = if prev_pressure.shape() == self.pressure_buffers[curr_idx].shape()
        {
            prev_pressure.to_owned()
        } else {
            self.pressure_buffers[prev_idx].clone()
        };

        // Check stability
        if !self.check_stability(dt, grid, medium, &pressure_current) {
            log::debug!("WesterveltWave: Potential instability at t={}", t);
        }

        // Get medium properties
        let rho_arr = medium.density_array();
        let c_arr = medium.sound_speed_array();
        // Get viscosity arrays directly from medium
        let eta_s_arr = medium.shear_viscosity_coeff_array();
        let eta_b_arr = medium.bulk_viscosity_coeff_array();
        // Create nonlinearity coefficient array
        let mut b_over_a_arr = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    b_over_a_arr[[i, j, k]] =
                        crate::domain::medium::AcousticProperties::nonlinearity_coefficient(
                            medium, x, y, z, grid,
                        );
                }
            }
        }

        // Compute Laplacian using spectral methods
        let start = Instant::now();
        let laplacian = if let Some(k_squared) = &self.k_squared {
            compute_laplacian_spectral(&pressure_current, k_squared)
        } else {
            compute_laplacian_fd(&pressure_current, grid)
        };

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_kspace(start.elapsed());
        }

        // Compute nonlinear term
        let start = Instant::now();
        let mut nonlinear_term = compute_nonlinear_term(
            &pressure_current,
            &pressure_previous,
            None, // Higher-order time history optional (2nd order sufficient per Hamilton & Blackstock 1998)
            medium,
            grid,
            dt,
        );

        // Apply nonlinearity scaling
        nonlinear_term *= self.nonlinearity_scaling;

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_nonlinear(start.elapsed());
        }

        // Compute damping/viscoelastic term
        let start = Instant::now();
        let damping_term = compute_viscoelastic_term(
            &pressure_current,
            &pressure_previous,
            &eta_s_arr,
            &eta_b_arr,
            &rho_arr,
            grid,
            dt,
        );

        {
            let mut metrics = self.metrics.lock().unwrap();
            // Record damping computation time as part of nonlinear time
            metrics.record_nonlinear(start.elapsed());
        }

        // Add source term
        let start = Instant::now();
        let src_mask = source.create_mask(grid);
        let src_amplitude = source.amplitude(t);
        let src_term = src_mask * src_amplitude;

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_source(start.elapsed());
        }

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_kspace(start.elapsed());
        }

        // Update pressure field using ndarray::Zip for better performance
        let start = Instant::now();
        let pressure_next = &mut self.pressure_buffers[next_idx];

        // Use parallel iteration for efficient computation
        use rayon::prelude::*;
        let dt2 = dt * dt;
        pressure_next
            .as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, p_next)| {
                let p_curr = pressure_current.as_slice().unwrap()[idx];
                let p_prev = pressure_previous.as_slice().unwrap()[idx];
                let c = c_arr.as_slice().unwrap()[idx];
                let lap = laplacian.as_slice().unwrap()[idx];
                let nl = nonlinear_term.as_slice().unwrap()[idx];
                let damp = damping_term.as_slice().unwrap()[idx];
                let src = src_term.as_slice().unwrap()[idx];

                let c2 = c * c;
                let update = dt2 * (c2 * lap + nl + damp + src);
                *p_next = 2.0 * p_curr - p_prev + update;
            });

        // Copy the result back to the fields array
        fields
            .index_axis_mut(Axis(0), UnifiedFieldType::Pressure.index())
            .assign(pressure_next);

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_combination(start.elapsed());
        }

        // Rotate buffer indices for next iteration
        self.buffer_indices.rotate_right(1);

        // Update step counters
        self.current_step += 1;
        self.current_time += dt;

        // Conservation diagnostics (if enabled)
        self.check_conservation_laws();

        Ok(())
    }

    fn report_performance(&self) {
        let metrics = self.metrics.lock().unwrap();
        metrics.summary();
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.nonlinearity_scaling = scaling;
    }
}

/// Check conservation laws and log diagnostics
///
/// Performs conservation checks at configured intervals and logs violations.
/// Critical violations trigger warnings via tracing infrastructure.
impl WesterveltWave {
    fn check_conservation_laws(&mut self) {
        // Check if we should perform diagnostics at this step
        let should_check = self.conservation_tracker.as_ref().is_some_and(|tracker| {
            self.current_step
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
            self.current_step,
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
                    eprintln!("⚠️  Westervelt Wave Conservation Warning: {}", diag);
                }
                ViolationSeverity::Error => {
                    eprintln!("❌ Westervelt Wave Conservation Error: {}", diag);
                }
                ViolationSeverity::Critical => {
                    eprintln!("🔴 Westervelt Wave Conservation CRITICAL: {}", diag);
                    eprintln!("   Solution may be physically invalid!");
                }
            }
        }
    }
}

/// Compute Laplacian using finite differences (fallback)
fn compute_laplacian_fd(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut laplacian = Array3::zeros((nx, ny, nz));

    let dx2_inv = 1.0 / (grid.dx * grid.dx);
    let dy2_inv = 1.0 / (grid.dy * grid.dy);
    let dz2_inv = 1.0 / (grid.dz * grid.dz);

    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                laplacian[[i, j, k]] = (field[[i + 1, j, k]] - 2.0 * field[[i, j, k]]
                    + field[[i - 1, j, k]])
                    * dx2_inv
                    + (field[[i, j + 1, k]] - 2.0 * field[[i, j, k]] + field[[i, j - 1, k]])
                        * dy2_inv
                    + (field[[i, j, k + 1]] - 2.0 * field[[i, j, k]] + field[[i, j, k - 1]])
                        * dz2_inv;
            }
        }
    }

    laplacian
}

/// Implementation of conservation diagnostics trait for Westervelt spectral solver
impl ConservationDiagnostics for WesterveltWave {
    fn calculate_total_energy(&self) -> f64 {
        // Acoustic energy density: E = p²/(2ρ₀c₀²)
        // Total energy: ∫∫∫ E dV
        let mut total_energy = 0.0;

        if let (Some(ref grid), Some(ref props)) = (&self.grid_cache, &self.medium_properties) {
            let factor = 1.0 / (2.0 * props.rho0 * props.c0 * props.c0);
            let dv = grid.dx * grid.dy * grid.dz; // Volume element

            // Use current pressure buffer
            let curr_idx = self.buffer_indices[1];
            let pressure = &self.pressure_buffers[curr_idx];

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let p = pressure[[i, j, k]];
                        total_energy += p * p * factor * dv;
                    }
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

        if let (Some(ref grid), Some(ref props)) = (&self.grid_cache, &self.medium_properties) {
            let dv = grid.dx * grid.dy * grid.dz;

            // Use current pressure buffer
            let curr_idx = self.buffer_indices[1];
            let pressure = &self.pressure_buffers[curr_idx];

            // Compute momentum from pressure gradients
            for i in 1..grid.nx - 1 {
                for j in 1..grid.ny - 1 {
                    for k in 1..grid.nz - 1 {
                        // Pressure gradients (central difference)
                        let dp_dx =
                            (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) / (2.0 * grid.dx);
                        let dp_dy =
                            (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) / (2.0 * grid.dy);
                        let dp_dz =
                            (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) / (2.0 * grid.dz);

                        // Momentum from pressure (acoustic approximation)
                        px += (props.rho0 * dp_dx / props.c0) * dv;
                        py += (props.rho0 * dp_dy / props.c0) * dv;
                        pz += (props.rho0 * dp_dz / props.c0) * dv;
                    }
                }
            }
        }

        (px, py, pz)
    }

    fn calculate_total_mass(&self) -> f64 {
        // For acoustic waves: ρ = ρ₀(1 + p/(ρ₀c₀²))
        // Total mass: ∫∫∫ ρ dV
        let mut total_mass = 0.0;

        if let (Some(ref grid), Some(ref props)) = (&self.grid_cache, &self.medium_properties) {
            let dv = grid.dx * grid.dy * grid.dz;

            // Use current pressure buffer
            let curr_idx = self.buffer_indices[1];
            let pressure = &self.pressure_buffers[curr_idx];

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let p = pressure[[i, j, k]];
                        let rho = props.rho0 * (1.0 + p / (props.rho0 * props.c0 * props.c0));
                        total_mass += rho * dv;
                    }
                }
            }
        }

        total_mass
    }
}
