//! Main Westervelt equation solver implementation

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use ndarray::{Array3, Array4, Axis, Zip};
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

    // Pressure history for second-order time derivatives
    pressure_history: Option<Array3<f64>>,
    prev_pressure_stored: Option<Array3<f64>>,

    // Performance tracking
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

impl WesterveltWave {
    /// Create a new Westervelt solver
    pub fn new(grid: &Grid) -> Self {
        let (k_squared, kx, ky, kz) = initialize_kspace_grids(grid);

        Self {
            k_squared: Some(k_squared),
            kx: Some(kx),
            ky: Some(ky),
            kz: Some(kz),
            nonlinearity_scaling: 1.0,
            max_pressure: 1e9,
            pressure_history: None,
            prev_pressure_stored: None,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
        }
    }

    /// Check numerical stability
    fn check_stability(
        &self,
        dt: f64,
        grid: &Grid,
        medium: &dyn Medium,
        pressure: &Array3<f64>,
    ) -> bool {
        let max_c = medium
            .sound_speed_array(grid)
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_c * dt / min_dx;

        let max_pressure = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

        cfl <= 0.5 && max_pressure < self.max_pressure
    }

    /// Update pressure history for second-order accuracy
    fn update_pressure_history(&mut self, current_pressure: &Array3<f64>) {
        if let Some(prev) = self.prev_pressure_stored.take() {
            self.pressure_history = Some(prev);
        }
        self.prev_pressure_stored = Some(current_pressure.clone());
    }

    /// Check if full second-order accuracy is available
    pub fn has_full_accuracy(&self) -> bool {
        self.pressure_history.is_some()
    }

    /// Get diagnostic information
    pub fn get_diagnostics(&self) -> String {
        let accuracy_status = if self.has_full_accuracy() {
            "Full second-order accuracy"
        } else {
            "Bootstrap initialization"
        };

        let metrics = self.metrics.lock().unwrap();
        format!(
            "WesterveltWave Diagnostics:\n\
             - Accuracy: {}\n\
             - Calls: {}\n\
             - Total time: {:.3}s",
            accuracy_status,
            metrics.call_count,
            metrics.total_time()
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

        // Get pressure field copy for calculations
        let pressure_current = fields
            .index_axis(Axis(0), UnifiedFieldType::Pressure.index())
            .to_owned();

        // Check stability
        if !self.check_stability(dt, grid, medium, &pressure_current) {
            log::debug!("WesterveltWave: Potential instability at t={}", t);
        }

        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if nx == 0 || ny == 0 || nz == 0 {
            return Ok(());
        }

        // Get medium properties
        let rho_arr = medium.density_array(grid);
        let c_arr = medium.sound_speed_array(grid);
        let eta_s_arr = medium.shear_viscosity_coeff_array();
        let eta_b_arr = medium.bulk_viscosity_coeff_array();

        // Source term
        let start = Instant::now();
        let mut src_term = Array3::zeros((nx, ny, nz));
        Zip::indexed(&mut src_term).for_each(|(i, j, k), val| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *val = source.get_source_term(t, x, y, z, grid);
        });
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_source(start.elapsed());
        }

        // Nonlinear term
        let start = Instant::now();
        let nonlinear_term = compute_nonlinear_term(
            &pressure_current,
            prev_pressure,
            self.pressure_history.as_ref(),
            medium,
            grid,
            dt,
        );
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_nonlinear(start.elapsed());
        }

        // Viscoelastic damping
        let damping_term = compute_viscoelastic_term(
            &pressure_current,
            prev_pressure,
            &eta_s_arr,
            &eta_b_arr,
            &rho_arr,
            grid,
            dt,
        );

        // Spectral Laplacian
        let start = Instant::now();
        let laplacian = if let Some(ref k_squared) = self.k_squared {
            compute_laplacian_spectral(&pressure_current, k_squared)
        } else {
            // Fallback to finite difference
            compute_laplacian_fd(&pressure_current, grid)
        };
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_kspace(start.elapsed());
        }

        // Update pressure field
        let start = Instant::now();
        let mut pressure_mut = fields.index_axis_mut(Axis(0), UnifiedFieldType::Pressure.index());

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = c_arr[[i, j, k]];
                    let c2 = c * c;
                    let lap = laplacian[[i, j, k]];
                    let nl = nonlinear_term[[i, j, k]];
                    let damp = damping_term[[i, j, k]];
                    let src = src_term[[i, j, k]];

                    let update = dt * dt * (c2 * lap + nl + damp + src);
                    pressure_mut[[i, j, k]] =
                        2.0 * pressure_current[[i, j, k]] - prev_pressure[[i, j, k]] + update;
                }
            }
        }

        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.record_combination(start.elapsed());
        }

        // Update pressure history with the newly computed pressure
        let updated_pressure = fields
            .index_axis(Axis(0), UnifiedFieldType::Pressure.index())
            .to_owned();
        self.update_pressure_history(&updated_pressure);

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
