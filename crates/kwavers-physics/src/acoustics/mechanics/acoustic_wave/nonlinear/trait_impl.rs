//! Trait implementations for `NonlinearWave`
//!
//! This module contains implementations of various traits for the `NonlinearWave` struct.

use crate::traits::AcousticWaveModel;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_source::Source;
use leto::{Array3, Array4};
use log::info;

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
        use kwavers_field::indices::PRESSURE_IDX;

        // Get a view of the current pressure field (avoid cloning)
        let pressure_view = fields
            .index_axis(0, PRESSURE_IDX)
            .expect("valid pressure axis index");

        // Create source term array
        let source_mask = source.create_mask(grid);
        let amplitude = source.amplitude(t);
        let mut source_term = Array3::zeros([grid.nx, grid.ny, grid.nz]);
        for (dst, &src) in source_term.iter_mut().zip(source_mask.iter()) {
            *dst = src * amplitude;
        }

        // Update using the nonlinear wave equation
        // Note: We pass a reference to avoid cloning, and the inner method is renamed
        let pressure_owned = pressure_view.to_contiguous();
        let updated_pressure = self.update_wave_inner(
            &pressure_owned,
            &source_term,
            medium,
            grid,
            (t / dt) as usize,
        )?;

        // Update the pressure field in the 4D array
        fields
            .index_axis_mut(0, PRESSURE_IDX)
            .map_err(|e| KwaversError::from(e.to_string()))?
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate_parameters(&self, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        // Check CFL condition
        if !self.is_stable(medium, grid) {
            return Err(kwavers_core::error::KwaversError::Physics(
                kwavers_core::error::PhysicsError::InvalidParameter {
                    parameter: "timestep".to_owned(),
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
            return Err(kwavers_core::error::KwaversError::Physics(
                kwavers_core::error::PhysicsError::InvalidParameter {
                    parameter: "sound_speed".to_owned(),
                    value: min_c,
                    reason: "Sound speed must be positive and finite".to_owned(),
                },
            ));
        }

        // Calculate minimum wavelength based on source frequency and minimum sound speed
        let min_wavelength = min_c / self.source_frequency;
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);

        // Ensure at least 6 points per wavelength for accurate simulation
        const MIN_POINTS_PER_WAVELENGTH: f64 = 6.0;
        if min_dx > min_wavelength / MIN_POINTS_PER_WAVELENGTH {
            return Err(kwavers_core::error::KwaversError::Physics(
                kwavers_core::error::PhysicsError::InvalidParameter {
                    parameter: "grid_spacing".to_owned(),
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
            return Err(kwavers_core::error::KwaversError::Physics(
                kwavers_core::error::PhysicsError::InvalidParameter {
                    parameter: "nonlinearity_scaling".to_owned(),
                    value: self.nonlinearity_scaling,
                    reason: "Must be between 0.0 and 10.0".to_owned(),
                },
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::wave_model::NonlinearWave;
    use crate::traits::AcousticWaveModel;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;

    /// `set_nonlinearity_scaling` stores the supplied value verbatim.
    #[test]
    fn set_nonlinearity_scaling_stores_supplied_value() {
        let dx = 0.001_f64;
        let grid = Grid::new(4, 4, 4, dx, dx, dx).unwrap();
        // CFL-safe dt for dx=1 mm, water: 0.9 * 0.001 / (π * 1500) ≈ 1.91e-7 s.
        let dt_init = 0.9 * dx / (std::f64::consts::PI * SOUND_SPEED_WATER_SIM);
        let mut w = NonlinearWave::new(&grid, dt_init);
        w.set_nonlinearity_scaling(3.7);
        assert!(
            (w.nonlinearity_scaling - 3.7).abs() < f64::EPSILON,
            "nonlinearity_scaling must be 3.7 (got {})",
            w.nonlinearity_scaling
        );
    }

    /// `validate_parameters` accepts a configuration that satisfies the CFL
    /// condition, grid-resolution criterion (≥6 pts/wavelength), and
    /// nonlinearity_scaling ∈ [0, 10].
    ///
    /// Parameters derived analytically for water (c≈1500 m/s, f=1 MHz):
    ///   λ = 1500/1e6 = 1.5 mm → min_dx < 1.5e-3/6 = 0.25 mm.
    ///   Use dx = 0.0001 m (0.1 mm) — factor-of-2.5 margin.
    ///   dt = get_stable_timestep → CFL = safety·threshold = 0.45 ≤ 0.45 ✓.
    #[test]
    fn validate_parameters_accepts_valid_config_for_water() {
        let dx = 0.0001_f64; // 0.1 mm
        let grid = Grid::new(10, 10, 10, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        // CFL-safe initial dt: PSTD formula dt = safety(0.9) * dx / (π * c).
        // Matches the NonlinearWave::compute_adaptive_timestep formula (stability.rs:61).
        let dt_init = 0.9 * dx / (std::f64::consts::PI * SOUND_SPEED_WATER_SIM);
        let mut w = NonlinearWave::new(&grid, dt_init);
        // Recompute from the actual medium to pick up medium-specific β/c corrections.
        w.dt = w.get_stable_timestep(&medium, &grid);
        w.source_frequency = MHZ_TO_HZ;

        assert!(
            w.validate_parameters(&medium, &grid).is_ok(),
            "analytically derived config must pass validate_parameters"
        );
    }

    /// `validate_parameters` rejects a timestep that violates the CFL condition.
    ///
    /// With water (c≈1500) and dx=0.001 m, CFL limit ≈ 0.45·dx/c ≈ 3e-7 s.
    /// dt=1e-3 s gives CFL≈1500 >> 0.45 → unstable.
    #[test]
    fn validate_parameters_rejects_unstable_dt() {
        let dx = 0.001_f64;
        let grid = Grid::new(4, 4, 4, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let w = NonlinearWave::new(&grid, 1e-3); // CFL >> 0.45

        let result = w.validate_parameters(&medium, &grid);
        assert!(result.is_err(), "CFL violation must return Err");
    }

    /// `validate_parameters` rejects a grid spacing too coarse for the source
    /// frequency.
    ///
    /// At f=5 MHz and c≈1500 m/s: λ=0.3 mm → min_dx must be < 0.3mm/6=50 μm.
    /// dx=0.001 m = 1 mm >> 50 μm → Err.
    #[test]
    fn validate_parameters_rejects_grid_too_coarse_for_frequency() {
        let dx = 0.001_f64; // 1 mm
        let grid = Grid::new(4, 4, 4, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        // Stable dt but source_frequency set so that wavelength requires finer grid
        let dt_safe = w.get_stable_timestep(&medium, &grid);
        w.dt = dt_safe;
        w.source_frequency = 5e6; // λ=0.3mm, need dx < 50µm; dx=1mm violates

        let result = w.validate_parameters(&medium, &grid);
        assert!(result.is_err(), "coarse grid must return Err");
    }

    /// `validate_parameters` rejects nonlinearity_scaling outside [0, 10].
    #[test]
    fn validate_parameters_rejects_out_of_range_nonlinearity_scaling() {
        let dx = 0.0001_f64;
        let grid = Grid::new(10, 10, 10, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        let dt_safe = w.get_stable_timestep(&medium, &grid);
        w.dt = dt_safe;
        w.source_frequency = MHZ_TO_HZ;

        // Negative scaling
        w.nonlinearity_scaling = -0.1;
        assert!(
            w.validate_parameters(&medium, &grid).is_err(),
            "negative nonlinearity_scaling must be rejected"
        );

        // Scaling exceeding upper bound
        w.nonlinearity_scaling = 10.1;
        assert!(
            w.validate_parameters(&medium, &grid).is_err(),
            "nonlinearity_scaling > 10.0 must be rejected"
        );
    }
}
