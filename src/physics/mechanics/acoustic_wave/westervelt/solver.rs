//! Main Westervelt equation solver implementation

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
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
    kx: Option<Array3<f64>>,
    ky: Option<Array3<f64>>,
    kz: Option<Array3<f64>>,

    // Configuration
    nonlinearity_scaling: f64,
    max_pressure: f64,

    // Pressure history using buffer rotation for zero-allocation updates
    pressure_buffers: [Array3<f64>; 3],
    buffer_indices: [usize; 3], // [next, current, previous]

    // Performance tracking
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl WesterveltWave {
    /// Create a new Westervelt solver
    pub fn new(grid: &Grid) -> Self {
        let (k_squared, kx, ky, kz) = initialize_kspace_grids(grid);
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            k_squared: Some(k_squared),
            kx: Some(kx),
            ky: Some(ky),
            kz: Some(kz),
            nonlinearity_scaling: 1.0,
            max_pressure: 1e6,
            pressure_buffers: [
                Array3::zeros(shape),
                Array3::zeros(shape),
                Array3::zeros(shape),
            ],
            buffer_indices: [0, 1, 2], // Initially: next=0, current=1, previous=2
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
        }
    }

    /// Check stability of the current configuration
    fn check_stability(
        &self,
        dt: f64,
        grid: &Grid,
        medium: &dyn Medium,
        pressure: &Array3<f64>,
    ) -> bool {
        // CFL condition check
        let max_c = medium
            .sound_speed_array(grid)
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x));
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_c * dt / min_dx;

        // Check pressure bounds
        let max_p = pressure.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));

        cfl < 0.5 && max_p < self.max_pressure
    }

    /// Initialize pressure buffers from the field
    fn initialize_buffers(&mut self, initial_pressure: &Array3<f64>) {
        self.pressure_buffers[self.buffer_indices[1]].assign(initial_pressure);
        self.pressure_buffers[self.buffer_indices[2]].assign(initial_pressure);
    }

    /// Get performance summary
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
        let rho_arr = medium.density_array(grid);
        let c_arr = medium.sound_speed_array(grid);
        // Get viscosity arrays from ElasticArrayAccess trait
        use crate::medium::elastic::ElasticArrayAccess;
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
                    b_over_a_arr[[i, j, k]] = medium.nonlinearity_coefficient(x, y, z, grid);
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
            None, // No pressure history for now
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
