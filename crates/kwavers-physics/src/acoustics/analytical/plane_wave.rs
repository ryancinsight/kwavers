//! Plane wave analytical solutions for validation

use super::utils::DISPERSION_CORRECTION_SECOND_ORDER;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use leto::Array3;

/// Plane wave analytical solutions
#[derive(Debug)]
pub struct PlaneWaveSolution;

impl PlaneWaveSolution {
    /// Generate analytical plane wave field
    pub fn generate(
        grid: &Grid,
        frequency: f64,
        amplitude: f64,
        sound_speed: f64,
        time: f64,
        direction: (f64, f64, f64),
    ) -> Array3<f64> {
        let mut field = Array3::zeros([grid.nx, grid.ny, grid.nz]);
        if !positive_finite(frequency)
            || !amplitude.is_finite()
            || !positive_finite(sound_speed)
            || !time.is_finite()
            || !positive_finite(grid.dx)
            || !positive_finite(grid.dy)
            || !positive_finite(grid.dz)
            || !direction.0.is_finite()
            || !direction.1.is_finite()
            || !direction.2.is_finite()
        {
            return field;
        }

        let wavelength = sound_speed / frequency;
        let k = TWO_PI / wavelength;
        let omega = TWO_PI * frequency;

        // Normalize direction vector
        let norm = direction
            .2
            .mul_add(
                direction.2,
                direction.1.mul_add(direction.1, direction.0.powi(2)),
            )
            .sqrt();
        if !positive_finite(norm) {
            return field;
        }

        let dir = (direction.0 / norm, direction.1 / norm, direction.2 / norm);

        // Apply dispersion correction for k-space methods
        let k_dispersed =
            k * (DISPERSION_CORRECTION_SECOND_ORDER * k * k * grid.dx).mul_add(grid.dx, 1.0);

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k_idx in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k_idx as f64 * grid.dz;

                    let phase =
                        k_dispersed * dir.2.mul_add(z, dir.0.mul_add(x, dir.1 * y)) - omega * time;
                    field[[i, j, k_idx]] = amplitude * phase.sin();
                }
            }
        }

        field
    }

    /// Validate plane wave propagation
    pub fn validate_propagation(
        initial_field: &Array3<f64>,
        final_field: &Array3<f64>,
        grid: &Grid,
        expected_speed: f64,
        time_elapsed: f64,
        tolerance: f64,
    ) -> bool {
        use super::utils::PhysicsTestUtils;

        let (actual_speed, correlation) = PhysicsTestUtils::detect_wave_propagation_subgrid(
            initial_field,
            final_field,
            grid,
            expected_speed,
            time_elapsed,
        );

        let speed_error = (actual_speed - expected_speed).abs() / expected_speed;
        speed_error < tolerance && correlation > 0.9
    }
}

#[inline]
fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use kwavers_grid::Grid;
    use std::f64::consts::PI;

    fn water_grid_one_wavelength() -> Grid {
        // f=1MHz, c=1500 m/s → λ=1.5mm; grid spacing dx=λ/32 → N=32 points per wavelength
        let f = MHZ_TO_HZ;
        let c = SOUND_SPEED_WATER_SIM;
        let lam = c / f;
        let dx = lam / 32.0;
        Grid::new(32, 4, 4, dx, dx, dx).unwrap()
    }

    /// Field at origin (i=j=k=0) at t=0 with x-direction propagation is 0.
    ///
    /// Phase = k_dispersed·(dir_x·0 + dir_y·0 + dir_z·0) − ω·0 = 0, sin(0) = 0.
    #[test]
    fn plane_wave_zero_at_origin_t0_x_direction() {
        let grid = water_grid_one_wavelength();
        let field = PlaneWaveSolution::generate(
            &grid,
            MHZ_TO_HZ,
            10.0,
            SOUND_SPEED_WATER_SIM,
            0.0,
            (1.0, 0.0, 0.0),
        );
        assert!(
            field[[0, 0, 0]].abs() < 1e-13,
            "field at origin must be 0 (got {:.3e})",
            field[[0, 0, 0]]
        );
    }

    /// Field is bounded by amplitude at every grid point.
    ///
    /// |sin(φ)| ≤ 1, so |field| ≤ amplitude·(1 + small dispersion correction).
    #[test]
    fn plane_wave_bounded_by_amplitude_everywhere() {
        let grid = water_grid_one_wavelength();
        let amplitude = 7.0_f64;
        let field = PlaneWaveSolution::generate(
            &grid,
            MHZ_TO_HZ,
            amplitude,
            SOUND_SPEED_WATER_SIM,
            0.0,
            (1.0, 0.0, 0.0),
        );
        for &v in field.iter() {
            assert!(
                v.abs() <= amplitude * 1.01,
                "field ({v:.3e}) must not exceed amplitude ({amplitude})"
            );
        }
    }

    /// Output array shape matches the grid shape exactly.
    #[test]
    fn plane_wave_shape_matches_grid() {
        let grid = water_grid_one_wavelength();
        let field = PlaneWaveSolution::generate(
            &grid,
            MHZ_TO_HZ,
            1.0,
            SOUND_SPEED_WATER_SIM,
            0.0,
            (1.0, 0.0, 0.0),
        );
        assert_eq!(field.shape(), [grid.nx, grid.ny, grid.nz]);
    }

    #[test]
    fn plane_wave_rejects_zero_direction_with_zero_field() {
        let grid = water_grid_one_wavelength();
        let field = PlaneWaveSolution::generate(
            &grid,
            MHZ_TO_HZ,
            1.0,
            SOUND_SPEED_WATER_SIM,
            0.0,
            (0.0, 0.0, 0.0),
        );

        assert_eq!(field.shape(), [grid.nx, grid.ny, grid.nz]);
        assert!(field.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn plane_wave_rejects_invalid_domains_with_zero_field() {
        let grid = water_grid_one_wavelength();

        let invalid_cases = [
            PlaneWaveSolution::generate(
                &grid,
                0.0,
                1.0,
                SOUND_SPEED_WATER_SIM,
                0.0,
                (1.0, 0.0, 0.0),
            ),
            PlaneWaveSolution::generate(
                &grid,
                MHZ_TO_HZ,
                f64::NAN,
                SOUND_SPEED_WATER_SIM,
                0.0,
                (1.0, 0.0, 0.0),
            ),
            PlaneWaveSolution::generate(
                &grid,
                MHZ_TO_HZ,
                1.0,
                -SOUND_SPEED_WATER_SIM,
                0.0,
                (1.0, 0.0, 0.0),
            ),
            PlaneWaveSolution::generate(
                &grid,
                MHZ_TO_HZ,
                1.0,
                SOUND_SPEED_WATER_SIM,
                f64::NAN,
                (1.0, 0.0, 0.0),
            ),
            PlaneWaveSolution::generate(
                &grid,
                MHZ_TO_HZ,
                1.0,
                SOUND_SPEED_WATER_SIM,
                0.0,
                (f64::INFINITY, 0.0, 0.0),
            ),
        ];

        for field in invalid_cases {
            assert_eq!(field.shape(), [grid.nx, grid.ny, grid.nz]);
            assert!(field.iter().all(|value| value.is_finite() && *value == 0.0));
        }
    }

    /// At i = N/4 (quarter wavelength) along x with t=0, the field is near peak.
    ///
    /// Analytical: k·(N/4·dx) ≈ (2π/λ)·(λ/4) = π/2 → sin(π/2) = 1.
    /// The dispersion-corrected wavenumber introduces a small positive correction:
    /// k_dispersed = k·(1 + 0.02·(k·dx)²) > k, so phase at i=N/4 > π/2
    /// and sin(phase) still ≈ 1 for large N.
    #[test]
    fn plane_wave_near_peak_at_quarter_wavelength() {
        let nx = 64usize;
        let f = MHZ_TO_HZ;
        let c = SOUND_SPEED_WATER_SIM;
        let lam = c / f;
        let dx = lam / nx as f64; // N grid points per wavelength
        let grid = Grid::new(nx, 4, 4, dx, dx, dx).unwrap();

        let amplitude = 5.0_f64;
        let field = PlaneWaveSolution::generate(&grid, f, amplitude, c, 0.0, (1.0, 0.0, 0.0));

        // At i = nx/4: k·x = (2π/λ)·(λ/4) = π/2
        let k = 2.0 * PI * f / c;
        let x_quarter = (nx / 4) as f64 * dx;

        // Dispersion correction: k_disp = k · (1 + 0.02 · (k·dx)²)
        let k_corr = k * (1.0 + DISPERSION_CORRECTION_SECOND_ORDER * (k * dx).powi(2));
        let actual_phase = k_corr * x_quarter;
        let expected_value = amplitude * actual_phase.sin();

        let got = field[[nx / 4, 0, 0]];
        assert!(
            (got - expected_value).abs() < 1e-12,
            "field at N/4 must match analytical sin (expected {expected_value:.6}, got {got:.6})"
        );
    }
}
