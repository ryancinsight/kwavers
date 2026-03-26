use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use log::debug;
use ndarray::Array3;
use std::time::Instant;

use super::super::wave_model::NonlinearWave;

impl NonlinearWave {
    /// Updates the wave field using the nonlinear acoustic wave equation.
    ///
    /// This method implements a pseudo-spectral time-domain (PSTD) solver for the
    /// nonlinear acoustic wave equation. It uses FFT for spatial derivatives and
    /// includes nonlinear terms for accurate modeling of high-intensity acoustics.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current pressure field \[Pa\]
    /// * `source` - Source term array [Pa/s²]
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    /// * `time_step` - Current time step index
    ///
    /// # Returns
    ///
    /// Updated pressure field (internal implementation)
    pub fn update_wave_inner(
        &mut self,
        pressure: &Array3<f64>,
        source: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<Array3<f64>> {
        let start_total = Instant::now();
        self.call_count += 1;

        // Validate inputs
        if pressure.shape() != [grid.nx, grid.ny, grid.nz] {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Pressure array shape [{}, {}, {}] doesn't match grid dimensions [{}, {}, {}]",
                pressure.shape()[0],
                pressure.shape()[1],
                pressure.shape()[2],
                grid.nx,
                grid.ny,
                grid.nz
            )));
        }

        // Compute nonlinear term
        let start_nonlinear = Instant::now();
        let nonlinear_term = self.compute_nonlinear_term(pressure, medium, grid)?;
        self.nonlinear_time += start_nonlinear.elapsed().as_secs_f64();

        // Apply k-space correction
        let start_fft = Instant::now();
        let linear_term = self.apply_k_space_correction(pressure, medium, grid)?;
        self.fft_time += start_fft.elapsed().as_secs_f64();

        // Add source term
        let start_source = Instant::now();
        let source_contribution = source * self.dt.powi(2);
        self.source_time += start_source.elapsed().as_secs_f64();

        // Combine terms
        let start_combination = Instant::now();
        let mut updated_pressure =
            linear_term + nonlinear_term * self.nonlinearity_scaling + source_contribution;

        // Apply stability constraints if needed
        if self.clamp_gradients {
            self.apply_stability_constraints(&mut updated_pressure);
        }

        self.combination_time += start_combination.elapsed().as_secs_f64();

        debug!(
            "Step {}: max pressure = {:.2e} Pa, update time = {:.3} ms",
            time_step,
            updated_pressure
                .iter()
                .fold(0.0_f64, |max, &val| max.max(val.abs())),
            start_total.elapsed().as_secs_f64() * 1000.0
        );

        Ok(updated_pressure)
    }
}
