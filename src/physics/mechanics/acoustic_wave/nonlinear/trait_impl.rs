//! Trait implementations for NonlinearWave
//!
//! This module contains implementations of various traits for the NonlinearWave struct.

use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::traits::AcousticWaveModel;
use crate::source::Source;
use crate::KwaversResult;
use log::info;
use ndarray::{Array3, Array4, Axis};

use super::wave_model::NonlinearWave;

impl AcousticWaveModel for NonlinearWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        // Extract pressure field from the 4D array (assuming index 0 is pressure)
        const PRESSURE_IDX: usize = 0;

        // Get a view of the current pressure field
        let pressure = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();

        // Create source term array
        let source_mask = source.create_mask(grid);
        let amplitude = source.amplitude(t);
        let source_term = source_mask * amplitude;

        // Update using the nonlinear wave equation
        match self.update_wave(&pressure, &source_term, medium, grid, (t / dt) as usize) {
            Ok(new_pressure) => {
                // Update the pressure field in the 4D array
                fields
                    .index_axis_mut(Axis(0), PRESSURE_IDX)
                    .assign(&new_pressure);
            }
            Err(e) => {
                log::error!("Error updating wave field: {:?}", e);
            }
        }
    }

    fn report_performance(&self) {
        info!("NonlinearWave Performance Metrics:");
        info!("  Total calls: {}", self.call_count);
        if self.call_count > 0 {
            info!("  Average times per call:");
            info!(
                "    Nonlinear term: {:.3} ms",
                self.nonlinear_time * 1000.0 / self.call_count as f64
            );
            info!(
                "    FFT operations: {:.3} ms",
                self.fft_time * 1000.0 / self.call_count as f64
            );
            info!(
                "    Source term: {:.3} ms",
                self.source_time * 1000.0 / self.call_count as f64
            );
            info!(
                "    Combination: {:.3} ms",
                self.combination_time * 1000.0 / self.call_count as f64
            );
            info!(
                "    Total: {:.3} ms",
                self.get_average_update_time() * 1000.0
            );
        }
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.nonlinearity_scaling = scaling;
    }
}

// Additional trait implementations for compatibility
impl NonlinearWave {
    /// Gets the stability timestep for the given medium and grid.
    pub fn get_stability_timestep(&self, medium: &dyn Medium, grid: &Grid) -> f64 {
        self.get_stable_timestep(medium, grid)
    }

    /// Validates the parameters for the simulation.
    pub fn validate_parameters(&self, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        // Check CFL condition
        if !self.is_stable(medium, grid) {
            return Err(crate::error::KwaversError::Physics(
                crate::error::PhysicsError::InvalidParameter {
                    parameter: "timestep".to_string(),
                    value: self.dt,
                    reason: format!(
                        "Must be <= {} for stability",
                        self.get_stable_timestep(medium, grid)
                    ),
                },
            ));
        }

        // Check grid resolution
        // Calculate minimum wavelength based on source frequency and sound speed
        let x = grid.nx as f64 * grid.dx / 2.0;
        let y = grid.ny as f64 * grid.dy / 2.0;
        let z = grid.nz as f64 * grid.dz / 2.0;
        let sound_speed = medium.sound_speed(x, y, z, grid);
        let min_wavelength = sound_speed / self.source_frequency;
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let points_per_wavelength = min_wavelength / min_dx;

        const MIN_POINTS_PER_WAVELENGTH: f64 = 4.0;
        if points_per_wavelength < MIN_POINTS_PER_WAVELENGTH {
            return Err(crate::error::KwaversError::Physics(
                crate::error::PhysicsError::InvalidParameter {
                    parameter: "grid_resolution".to_string(),
                    value: points_per_wavelength,
                    reason: format!(
                        "Minimum {:.1} points/wavelength required, got {:.1}",
                        MIN_POINTS_PER_WAVELENGTH, points_per_wavelength
                    ),
                },
            ));
        }

        // Validate multi-frequency configuration if present
        if let Some(ref config) = self.multi_freq_config {
            if !config.validate() {
                return Err(crate::error::KwaversError::Config(
                    crate::error::ConfigError::InvalidValue {
                        parameter: "multi_frequency".to_string(),
                        value: "invalid".to_string(),
                        constraint: "Configuration must be valid".to_string(),
                    },
                ));
            }
        }

        Ok(())
    }

    /// Gets the model name.
    pub fn get_model_name(&self) -> &str {
        "NonlinearWave"
    }

    /// Checks if the model supports heterogeneous media.
    pub fn supports_heterogeneous_media(&self) -> bool {
        // PSTD has limitations with strongly heterogeneous media
        // See documentation in wave_model.rs
        true // But with caveats
    }

    /// Gets the required number of ghost cells.
    pub fn get_required_ghost_cells(&self) -> usize {
        // PSTD doesn't require ghost cells as it uses global FFTs
        0
    }
}
