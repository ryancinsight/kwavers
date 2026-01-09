//! Trait implementations for `NonlinearWave`
//!
//! This module contains implementations of various traits for the `NonlinearWave` struct.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::Source;
use crate::physics::traits::AcousticWaveModel;
use log::info;
use ndarray::{Array3, Array4, Axis};

use super::wave_model::NonlinearWave;

impl AcousticWaveModel for NonlinearWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        // Extract pressure field from the 4D array
        use crate::domain::field::indices::PRESSURE_IDX;

        // Get a view of the current pressure field (avoid cloning)
        let pressure_view = fields.index_axis(Axis(0), PRESSURE_IDX);

        // Create source term array
        let source_mask = source.create_mask(grid);
        let amplitude = source.amplitude(t);
        let source_term = source_mask * amplitude;

        // Update using the nonlinear wave equation
        // Note: We pass a reference to avoid cloning, and the inner method is renamed
        let updated_pressure = self.update_wave_inner(
            &pressure_view.to_owned(),
            &source_term,
            medium,
            grid,
            (t / dt) as usize,
        )?;

        // Update the pressure field in the 4D array
        fields
            .index_axis_mut(Axis(0), PRESSURE_IDX)
            .assign(&updated_pressure);

        Ok(())
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

impl NonlinearWave {
    /// Validates the parameters for the simulation using the minimum sound speed.
    pub fn validate_parameters(&self, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        // Check CFL condition
        if !self.is_stable(medium, grid) {
            return Err(crate::core::error::KwaversError::Physics(
                crate::core::error::PhysicsError::InvalidParameter {
                    parameter: "timestep".to_string(),
                    value: self.dt,
                    reason: format!(
                        "Must be <= {} for stability",
                        self.get_stable_timestep(medium, grid)
                    ),
                },
            ));
        }

        // Check grid resolution using minimum sound speed for heterogeneous media
        // Get the sound speed array and find the minimum
        let c_array = medium.sound_speed_array();
        let min_c = c_array.iter().fold(
            f64::INFINITY,
            |acc, &x| if x > 0.0 { acc.min(x) } else { acc },
        );

        if min_c <= 0.0 || min_c.is_infinite() {
            return Err(crate::core::error::KwaversError::Physics(
                crate::core::error::PhysicsError::InvalidParameter {
                    parameter: "sound_speed".to_string(),
                    value: min_c,
                    reason: "Sound speed must be positive and finite".to_string(),
                },
            ));
        }

        // Calculate minimum wavelength based on source frequency and minimum sound speed
        let min_wavelength = min_c / self.source_frequency;
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);

        // Ensure at least 6 points per wavelength for accurate simulation
        const MIN_POINTS_PER_WAVELENGTH: f64 = 6.0;
        if min_dx > min_wavelength / MIN_POINTS_PER_WAVELENGTH {
            return Err(crate::core::error::KwaversError::Physics(
                crate::core::error::PhysicsError::InvalidParameter {
                    parameter: "grid_spacing".to_string(),
                    value: min_dx,
                    reason: format!(
                        "Grid spacing too large. Maximum allowed: {} m for minimum wavelength {} m at frequency {} Hz with minimum sound speed {} m/s",
                        min_wavelength / MIN_POINTS_PER_WAVELENGTH,
                        min_wavelength,
                        self.source_frequency,
                        min_c
                    ),
                },
            ));
        }

        // Check for valid nonlinearity scaling
        if self.nonlinearity_scaling < 0.0 || self.nonlinearity_scaling > 10.0 {
            return Err(crate::core::error::KwaversError::Physics(
                crate::core::error::PhysicsError::InvalidParameter {
                    parameter: "nonlinearity_scaling".to_string(),
                    value: self.nonlinearity_scaling,
                    reason: "Must be between 0.0 and 10.0".to_string(),
                },
            ));
        }

        Ok(())
    }
}
