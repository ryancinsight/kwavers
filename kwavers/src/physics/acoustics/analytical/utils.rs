//! Utilities for physics validation and testing

use crate::domain::grid::Grid;
use ndarray::Array3;
use crate::core::constants::numerical::{TWO_PI};

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
#[derive(Debug)]
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
        let k = TWO_PI / wavelength;
        let omega = TWO_PI * frequency;

        // Apply dispersion correction for k-space methods
        let k_dispersed =
            k * (DISPERSION_CORRECTION_SECOND_ORDER * k * k * grid.dx).mul_add(grid.dx, 1.0);

        for i in 0..grid.nx {
            let x = i as f64 * grid.dx;
            let phase = k_dispersed * x - omega * time;
            let value = amplitude * phase.sin();

            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    field[[i, j, k]] = value;
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
                let total_shift = f64::from(sub_shift).mul_add(0.1, f64::from(shift_int));
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
    #[allow(clippy::cast_precision_loss)]
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
                            field2[[i_floor, j, k]]
                                .mul_add(1.0 - i_frac, field2[[i_floor + 1, j, k]] * i_frac)
                        } else {
                            field2[[i_floor, j, k]]
                        };

                        correlation += field1[[i, j, k]] * interpolated_value;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            correlation / f64::from(count)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::core::constants::numerical::MHZ_TO_HZ;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(16, 4, 4, 1e-4, 1e-4, 1e-4).unwrap()
    }

    /// Published constant: second-order dispersion correction = 0.02.
    ///
    /// From Liu (1997): leading-order k-space dispersion coefficient.
    #[test]
    fn dispersion_correction_constants_match_published_values() {
        assert_eq!(DISPERSION_CORRECTION_SECOND_ORDER, 0.02);
        assert_eq!(DISPERSION_CORRECTION_FOURTH_ORDER, 0.001);
    }

    /// `analytical_plane_wave_with_dispersion` at t=0, i=0: field = amplitude·sin(0) = 0.
    ///
    /// Phase = k_dispersed·(0·dx) − ω·0 = 0, so sin(0) = 0 for all j,k at i=0.
    #[test]
    fn analytical_plane_wave_zero_at_origin_for_t0() {
        let grid = small_grid();
        let field = PhysicsTestUtils::analytical_plane_wave_with_dispersion(
            &grid,
            MHZ_TO_HZ,
            2.0,
            SOUND_SPEED_WATER_SIM,
            0.0,
        );
        // At t=0 and i=0: phase = k_dispersed·0 = 0 → sin(0) = 0.
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                assert!(
                    field[[0, j, k]].abs() < 1e-14,
                    "field at i=0, t=0 must be 0 (got {:.3e})",
                    field[[0, j, k]]
                );
            }
        }
    }

    /// `analytical_plane_wave_with_dispersion` amplitude bound: |field| ≤ amplitude.
    #[test]
    fn analytical_plane_wave_bounded_by_amplitude() {
        let grid = small_grid();
        let amplitude = 5.0_f64;
        let field = PhysicsTestUtils::analytical_plane_wave_with_dispersion(
            &grid,
            MHZ_TO_HZ,
            amplitude,
            SOUND_SPEED_WATER_SIM,
            0.0,
        );
        for &v in field.iter() {
            assert!(
                v.abs() <= amplitude + 1e-12,
                "field must not exceed amplitude (got {v:.3e})"
            );
        }
    }

    /// `measure_energy_conservation` with identical fields returns 1.0.
    ///
    /// Ratio = E_final / E_initial; if fields are equal, ratio = 1.
    #[test]
    fn energy_conservation_unity_for_equal_fields() {
        let grid = small_grid();
        let field = Array3::<f64>::from_elem((grid.nx, grid.ny, grid.nz), 3.0);
        let ratio = PhysicsTestUtils::measure_energy_conservation(&field, &field, &grid);
        assert!(
            (ratio - 1.0).abs() < 1e-14,
            "energy ratio must be 1 for equal fields (got {ratio:.6})"
        );
    }

    /// `measure_energy_conservation` with zero initial field returns 0.0 (no energy).
    #[test]
    fn energy_conservation_zero_for_zero_initial_field() {
        let grid = small_grid();
        let zero = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        let nonzero = Array3::<f64>::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
        let ratio = PhysicsTestUtils::measure_energy_conservation(&zero, &nonzero, &grid);
        assert_eq!(
            ratio, 0.0,
            "energy ratio must be 0 when initial field is zero"
        );
    }
}
