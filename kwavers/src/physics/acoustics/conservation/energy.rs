//! Acoustic energy conservation checks.

use crate::domain::grid::Grid;
use ndarray::{Array3, Zip};

/// Compute total acoustic energy and relative error against `initial_energy`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn validate_energy_conservation(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    initial_energy: f64,
    grid: &Grid,
) -> f64 {
    let mut total_energy = 0.0_f64;
    let dv = grid.dx * grid.dy * grid.dz;

    Zip::from(pressure)
        .and(velocity_x)
        .and(velocity_y)
        .and(velocity_z)
        .and(density)
        .and(sound_speed)
        .for_each(|&p, &vx, &vy, &vz, &rho, &c| {
            if rho > 0.0 && c > 0.0 {
                let kinetic = 0.5 * rho * vz.mul_add(vz, vx.mul_add(vx, vy * vy));
                let potential = p * p / (2.0 * rho * c * c);
                total_energy += (kinetic + potential) * dv;
            }
        });

    (total_energy - initial_energy).abs() / initial_energy.max(1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap()
    }

    fn make_fields(
        p: f64,
        v: f64,
        rho: f64,
        c: f64,
    ) -> (
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
    ) {
        let s = (4, 4, 4);
        let pressure = Array3::from_elem(s, p);
        let velocity_x = Array3::from_elem(s, v);
        let velocity_y = Array3::zeros(s);
        let velocity_z = Array3::zeros(s);
        let density = Array3::from_elem(s, rho);
        let sound_speed = Array3::from_elem(s, c);
        (
            pressure,
            velocity_x,
            velocity_y,
            velocity_z,
            density,
            sound_speed,
        )
    }

    /// When `initial_energy` equals the computed total, relative error = 0.
    #[test]
    fn energy_conservation_error_zero_when_initial_matches_computed() {
        let grid = small_grid();
        let (p, vx, vy, vz, rho, c) = make_fields(100.0, 0.1, 1000.0, SOUND_SPEED_WATER_SIM);

        // Compute what the function computes first, then pass it as initial
        // so the error is exactly 0.
        let dv = grid.dx * grid.dy * grid.dz;
        let p_val = 100.0_f64;
        let v_val = 0.1_f64;
        let rho_val = 1000.0_f64;
        let c_val = SOUND_SPEED_WATER_SIM;
        let kinetic = 0.5 * rho_val * v_val * v_val;
        let potential = p_val * p_val / (2.0 * rho_val * c_val * c_val);
        let cell_energy = (kinetic + potential) * dv;
        let initial_energy = cell_energy * (4.0_f64).powi(3);

        let error =
            validate_energy_conservation(&p, &vx, &vy, &vz, &rho, &c, initial_energy, &grid);
        assert!(error.abs() < 1e-12, "error must be 0 (got {error:.3e})");
    }

    /// All-zero pressure/velocity: total energy = 0; error = initial/(max(initial, 1e-10)).
    /// With initial=0: denominator = 1e-10, numerator = 0 → error = 0.
    #[test]
    fn energy_conservation_zero_error_for_zero_fields_and_zero_initial() {
        let grid = small_grid();
        let (p, vx, vy, vz, rho, c) = make_fields(0.0, 0.0, 1000.0, SOUND_SPEED_WATER_SIM);
        let error = validate_energy_conservation(&p, &vx, &vy, &vz, &rho, &c, 0.0, &grid);
        assert_eq!(
            error, 0.0,
            "zero fields with zero initial energy must give 0 error"
        );
    }
}
