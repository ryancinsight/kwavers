//! Entropy-production checks for acoustic absorption.

use kwavers_grid::Grid;
use ndarray::Array3;

/// Compute volumetric irreversible entropy production rate [W/K].
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn entropy_production_rate(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
    temperature: f64,
    grid: &Grid,
) -> f64 {
    debug_assert!(temperature > 0.0, "Temperature must be positive Kelvin");
    if temperature <= 0.0 {
        // Defensive runtime guard: entropy production is undefined at T ≤ 0 K.
        // Returning 0 is conservative; the debug_assert above flags caller bugs.
        // Prior to 2026-05-21 this clamped t0_inv = 1/max(T, 1.0), silently
        // corrupting cryogenic-temperature inputs (T < 1 K) by replacing T
        // with 1 K rather than reporting the true 1/T scaling.
        return 0.0;
    }
    let dv = grid.dx * grid.dy * grid.dz;
    let t0_inv = 1.0 / temperature;
    let mut total = 0.0_f64;
    let (nx, ny, nz) = pressure.dim();

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                let alpha = absorption[[i, j, k]];
                if rho > 0.0 && c > 0.0 {
                    let p = pressure[[i, j, k]];
                    let vx = velocity_x[[i, j, k]];
                    let vy = velocity_y[[i, j, k]];
                    let vz = velocity_z[[i, j, k]];
                    let energy_density = (0.5 * rho).mul_add(
                        vz.mul_add(vz, vx.mul_add(vx, vy * vy)),
                        p * p / (2.0 * rho * c * c),
                    );
                    total += 2.0 * alpha * c * energy_density * t0_inv * dv;
                }
            }
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
    use kwavers_grid::Grid;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap()
    }

    fn uniform(s: (usize, usize, usize), val: f64) -> Array3<f64> {
        Array3::from_elem(s, val)
    }

    /// Zero acoustic fields (p=0, v=0) with non-zero absorption → zero entropy production.
    ///
    /// energy_density = 0 → 2·α·c·0/T = 0 regardless of α.
    #[test]
    fn entropy_production_zero_for_zero_energy_fields() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let zero = Array3::<f64>::zeros(s);
        let rho = uniform(s, DENSITY_WATER_NOMINAL);
        let c = uniform(s, SOUND_SPEED_WATER_SIM);
        let alpha = uniform(s, 5.0); // non-zero absorption, but zero energy
        let ds = entropy_production_rate(
            &zero,
            &zero,
            &zero,
            &zero,
            &rho,
            &c,
            &alpha,
            BODY_TEMPERATURE_K,
            &grid,
        );
        assert_eq!(
            ds, 0.0,
            "zero acoustic energy must give zero entropy production"
        );
    }

    /// Analytical formula: ds/dt = α·P²/(ρ·c·T)·V for pressure-only field.
    ///
    /// Derivation: energy_density = P²/(2ρc²), ds/dt = 2·α·c·P²/(2ρc²·T)·N·dV = α·P²·N·dV/(ρ·c·T).
    #[test]
    fn entropy_production_matches_analytical_formula_pressure_field() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let p0 = 3000.0_f64;
        let rho0 = DENSITY_WATER_NOMINAL;
        let c0 = SOUND_SPEED_WATER_SIM;
        let a0 = 2.0_f64;
        let t0 = BODY_TEMPERATURE_K;
        let p = uniform(s, p0);
        let v = Array3::<f64>::zeros(s);
        let rho = uniform(s, rho0);
        let c = uniform(s, c0);
        let alpha = uniform(s, a0);
        let vol = grid.dx * grid.dy * grid.dz * (s.0 * s.1 * s.2) as f64;

        let ds = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha, t0, &grid);

        let expected = a0 * p0 * p0 / (rho0 * c0 * t0) * vol;
        let rel = (ds - expected).abs() / expected;
        assert!(
            rel < 1e-10,
            "entropy rel-error={rel:.3e}, expected={expected:.6e}, got={ds:.6e}"
        );
    }
}
