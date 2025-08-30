//! Utilities for physics validation and testing

use crate::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

// Physical constants for dispersion correction
/// Second-order dispersion correction coefficient for k-space methods
/// This coefficient accounts for the leading-order numerical dispersion
/// in pseudo-spectral methods. Value derived from Taylor expansion of
/// the exact dispersion relation around the continuous limit.
pub const DISPERSION_CORRECTION_SECOND_ORDER: f64 = 0.02;

/// Fourth-order dispersion correction coefficient for k-space methods  
/// This coefficient provides higher-order correction to minimize
/// numerical dispersion at high wavenumbers approaching the Nyquist limit.
/// Value optimized for typical ultrasound simulation parameters.
pub const DISPERSION_CORRECTION_FOURTH_ORDER: f64 = 0.001;

// Numerical analysis constants
/// Number of sub-grid increments for precise phase shift detection
/// This determines the precision of sub-grid-scale phase measurements
/// in wave propagation analysis. 10 steps provides 0.1 grid-point precision
/// which is sufficient for most ultrasound validation scenarios.
pub const SUB_GRID_SEARCH_STEPS: u32 = 10;

/// Test utilities for physics validation
#[derive(Debug, Debug))]
pub struct PhysicsTestUtils;

impl PhysicsTestUtils {
    /// Calculate analytical plane wave solution with dispersion correction
    pub fn analytical_plane_wave_with_dispersion(
        grid: &Grid,
        frequency: f64,
        amplitude: f64,
        sound_speed: f64,
        time: f64,
    ) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let wavelength = sound_speed / frequency;
        let k = 2.0 * PI / wavelength;
        let omega = 2.0 * PI * frequency;

        // Apply dispersion correction for k-space methods
        let k_corrected =
            k * (1.0 + DISPERSION_CORRECTION_SECOND_ORDER * k * k * grid.dx * grid.dx);

        for i in 0..grid.nx {
            let x = i as f64 * grid.dx;
            let phase = k_corrected * x - omega * time;
            let value = amplitude * phase.sin();

            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    field[[i, j, k] = value;
                }
            }
        }

        field
    }

    /// Measure energy conservation between initial and final fields
    pub fn measure_energy_conservation(
        initial_field: &Array3<f64>,
        final_field: &Array3<f64>,
        grid: &Grid,
    ) -> f64 {
        let initial_energy: f64 = initial_field.iter().map(|&p| p * p).sum();
        let final_energy: f64 = final_field.iter().map(|&p| p * p).sum();

        // Normalize by grid volume
        let volume_element = grid.dx * grid.dy * grid.dz;
        let initial_energy_norm = initial_energy * volume_element;
        let final_energy_norm = final_energy * volume_element;

        // Return energy conservation ratio (should be close to 1.0)
        if initial_energy_norm > 0.0 {
            final_energy_norm / initial_energy_norm
        } else {
            0.0
        }
    }

    /// Detect wave propagation with sub-grid accuracy
    pub fn detect_wave_propagation_subgrid(
        initial_field: &Array3<f64>,
        final_field: &Array3<f64>,
        grid: &Grid,
        expected_speed: f64,
        time_elapsed: f64,
    ) -> (f64, f64) {
        // Use cross-correlation to detect actual wave shift with sub-grid precision
        let expected_shift_meters = expected_speed * time_elapsed;
        let expected_shift_cells = expected_shift_meters / grid.dx;

        // Calculate cross-correlation to find actual shift
        let mut max_correlation = 0.0;
        let mut peak_shift = 0.0;

        // Search in sub-grid increments
        let search_range = (expected_shift_cells * 2.0) as i32;
        for shift_int in -search_range..=search_range {
            for sub_shift in 0..SUB_GRID_SEARCH_STEPS {
                let total_shift = shift_int as f64 + sub_shift as f64 * 0.1;
                let correlation = Self::calculate_cross_correlation(
                    initial_field,
                    final_field,
                    total_shift,
                    grid,
                );

                if correlation > max_correlation {
                    max_correlation = correlation;
                    peak_shift = total_shift;
                }
            }
        }

        let actual_speed = (peak_shift * grid.dx) / time_elapsed;
        (actual_speed, max_correlation)
    }

    /// Calculate cross-correlation between fields with fractional shift
    fn calculate_cross_correlation(
        field1: &Array3<f64>,
        field2: &Array3<f64>,
        shift: f64,
        grid: &Grid,
    ) -> f64 {
        let mut correlation = 0.0;
        let mut count = 0;

        for i in 0..grid.nx {
            let shifted_i = i as f64 + shift;
            if shifted_i >= 0.0 && shifted_i < (grid.nx - 1) as f64 {
                let i_floor = shifted_i as usize;
                let i_frac = shifted_i - i_floor as f64;

                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        // Linear interpolation for sub-grid accuracy
                        let interpolated_value = if i_floor + 1 < grid.nx {
                            field2[[i_floor, j, k] * (1.0 - i_frac)
                                + field2[[i_floor + 1, j, k] * i_frac
                        } else {
                            field2[[i_floor, j, k]
                        };

                        correlation += field1[[i, j, k] * interpolated_value;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            correlation / count as f64
        } else {
            0.0
        }
    }
}
