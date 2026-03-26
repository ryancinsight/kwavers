use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
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
        field.iter_mut().for_each(|val| {
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
                    let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
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
