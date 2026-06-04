use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use log::warn;
use ndarray::Array3;
use std::f64;

use super::super::wave_model::NonlinearWave;

impl NonlinearWave {
    /// Applies stability constraints to prevent numerical instabilities.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to constrain (modified in place)
    pub(crate) fn apply_stability_constraints(&self, field: &mut Array3<f64>) {
        // Clamp extreme values
        use rayon::prelude::*;
        field.par_iter_mut().for_each(|val| {
            if val.abs() > self.max_pressure {
                *val = val.signum() * self.max_pressure;
            }
            // Remove NaN or Inf values
            if !val.is_finite() {
                *val = 0.0;
                warn!("Non-finite value detected and zeroed in pressure field");
            }
        });
    }

    /// Computes adaptive time step based on CFL condition.
    ///
    /// # Arguments
    ///
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Recommended time step \[s\]
    pub fn compute_adaptive_timestep(&self, medium: &dyn Medium, grid: &Grid) -> f64 {
        // Get actual maximum sound speed from medium
        let mut max_c: f64 = 0.0;
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let c = kwavers_domain::medium::sound_speed_at(medium, x, y, z, grid);
                    max_c = max_c.max(c);
                }
            }
        }

        // Fall back to stored value if medium returns zero
        if max_c <= 0.0 {
            max_c = self.max_sound_speed;
        }

        let min_dx = grid.dx.min(grid.dy).min(grid.dz);

        // CFL condition for PSTD
        let dt_cfl = self.cfl_safety_factor * min_dx / (f64::consts::PI * max_c);

        // Additional constraint for nonlinear terms
        let dt_nonlinear = if self.nonlinearity_scaling > 0.0 {
            // Get typical B/A from center of grid
            let cx = grid.nx / 2;
            let cy = grid.ny / 2;
            let cz = grid.nz / 2;
            let (x, y, z) = grid.indices_to_coordinates(cx, cy, cz);
            let beta = medium.nonlinearity_coefficient(x, y, z, grid);
            min_dx / (beta * max_c)
        } else {
            f64::INFINITY
        };

        dt_cfl.min(dt_nonlinear)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::wave_model::NonlinearWave;
    use kwavers_grid::Grid;
    use kwavers_domain::medium::HomogeneousMedium;
    use ndarray::Array3;

    /// Values beyond `max_pressure` are clamped to ±max_pressure.
    #[test]
    fn apply_stability_constraints_clamps_values_exceeding_max_pressure() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.max_pressure = 1_000.0; // Override to a known limit

        let mut field = Array3::<f64>::from_elem((4, 4, 4), 5_000.0); // 5× limit
        w.apply_stability_constraints(&mut field);

        for &v in field.iter() {
            assert_eq!(v, 1_000.0, "all values must be clamped to max_pressure");
        }
    }

    /// Negative values beyond −max_pressure are clamped to −max_pressure.
    #[test]
    fn apply_stability_constraints_clamps_negative_values() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.max_pressure = 1_000.0;

        let mut field = Array3::<f64>::from_elem((4, 4, 4), -5_000.0);
        w.apply_stability_constraints(&mut field);

        for &v in field.iter() {
            assert_eq!(v, -1_000.0, "negative excess must clamp to -max_pressure");
        }
    }

    /// NaN is not caught by the `> max_pressure` branch (NaN comparisons are false)
    /// but is caught by the `!is_finite()` branch and zeroed.
    /// ±Inf is caught by the clamp branch first (Inf > max_pressure is true),
    /// becoming ±max_pressure which is finite.
    #[test]
    fn apply_stability_constraints_zeroes_nan_and_clamps_inf() {
        let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.max_pressure = 1e8;

        let mut field = Array3::<f64>::zeros((4, 4, 4));
        field[[0, 0, 0]] = f64::NAN;
        field[[1, 1, 1]] = f64::INFINITY;
        field[[2, 2, 2]] = f64::NEG_INFINITY;

        w.apply_stability_constraints(&mut field);

        // NaN: NaN.abs() > 1e8 is false (NaN comparisons always false);
        // !NaN.is_finite() = true → zeroed.
        assert_eq!(field[[0, 0, 0]], 0.0, "NaN must be zeroed");
        // Inf: Inf.abs() > 1e8 → clamped to signum(Inf)·1e8 = +1e8; is_finite() → stays.
        assert_eq!(
            field[[1, 1, 1]],
            1e8,
            "Inf must be clamped to +max_pressure"
        );
        // -Inf: similar → -1e8.
        assert_eq!(
            field[[2, 2, 2]],
            -1e8,
            "NegInf must be clamped to -max_pressure"
        );
    }

    /// `compute_adaptive_timestep` returns a positive, finite time step for water.
    /// Analytical lower bound: dt_cfl = safety·min_dx/(π·c) > 0.
    #[test]
    fn compute_adaptive_timestep_returns_positive_finite_value_for_water() {
        let dx = 0.001_f64;
        let grid = Grid::new(4, 4, 4, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let w = NonlinearWave::new(&grid, 1e-7);

        let dt = w.compute_adaptive_timestep(&medium, &grid);
        assert!(
            dt > 0.0,
            "adaptive timestep must be positive (got {dt:.3e})"
        );
        assert!(dt.is_finite(), "adaptive timestep must be finite");
        // For water c≈1500, safety=0.9: dt_cfl = 0.9·0.001/(π·1500) ≈ 1.91e-7
        assert!(
            dt < 1e-5,
            "adaptive timestep must be sub-microsecond for water/mm grid"
        );
    }
}
