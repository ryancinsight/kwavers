//! Aggregated conservation validation.

use super::{
    entropy_production_rate, validate_energy_conservation, validate_mass_conservation,
    validate_momentum_conservation, AcousticConservationMetrics, AcousticStateRefs,
    ConservationParams, PreviousFields,
};
use crate::domain::grid::Grid;

/// Run all conservation checks and return consolidated metrics.
#[must_use]
pub fn validate_conservation(
    state: AcousticStateRefs<'_>,
    prev: PreviousFields<'_>,
    params: ConservationParams,
    grid: &Grid,
) -> AcousticConservationMetrics {
    let energy_error = validate_energy_conservation(
        state.pressure,
        state.velocity_x,
        state.velocity_y,
        state.velocity_z,
        state.density,
        state.sound_speed,
        params.initial_energy,
        grid,
    );
    let mass_error = if let Some(rho_prev) = prev.density {
        validate_mass_conservation(
            state.density,
            rho_prev,
            state.velocity_x,
            state.velocity_y,
            state.velocity_z,
            params.dt,
            grid,
        )
    } else {
        0.0
    };
    let momentum_error = if let Some(v_prev) = prev.velocity {
        validate_momentum_conservation(
            state.velocity_x,
            state.velocity_y,
            state.velocity_z,
            v_prev.x,
            v_prev.y,
            v_prev.z,
            state.pressure,
            state.density,
            params.dt,
            grid,
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    let ds_dt = entropy_production_rate(
        state.pressure,
        state.velocity_x,
        state.velocity_y,
        state.velocity_z,
        state.density,
        state.sound_speed,
        state.absorption,
        params.temperature,
        grid,
    );
    let is_conserved = energy_error < params.tolerance
        && mass_error < params.tolerance
        && momentum_error.0 < params.tolerance
        && momentum_error.1 < params.tolerance
        && momentum_error.2 < params.tolerance
        && ds_dt >= 0.0;

    AcousticConservationMetrics {
        energy_error,
        mass_error,
        momentum_error,
        entropy_production_rate: ds_dt,
        is_conserved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    fn small_grid() -> Grid {
        Grid::new(6, 6, 6, 1e-3, 1e-3, 1e-3).unwrap()
    }

    fn uniform(s: (usize, usize, usize), val: f64) -> Array3<f64> {
        Array3::from_elem(s, val)
    }

    /// All `None` previous-step fields → mass_error = 0.0, momentum_error = (0,0,0).
    ///
    /// The aggregator short-circuits to 0 when optional history is absent,
    /// supporting first-timestep invocation without prior state.
    #[test]
    fn validate_conservation_none_previous_gives_zero_mass_and_momentum() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let p = uniform(s, 1000.0);
        let v = uniform(s, 0.0);
        let rho = uniform(s, 1000.0);
        let c = uniform(s, 1500.0);
        let alpha = uniform(s, 0.0);
        let dv = grid.dx * grid.dy * grid.dz;
        let n = (s.0 * s.1 * s.2) as f64;
        let init = 1000.0_f64.powi(2) / (2.0 * 1000.0 * 1500.0_f64.powi(2)) * dv * n;

        let state = AcousticStateRefs {
            pressure: &p,
            velocity_x: &v,
            velocity_y: &v,
            velocity_z: &v,
            density: &rho,
            sound_speed: &c,
            absorption: &alpha,
        };
        let prev = PreviousFields {
            pressure: None,
            velocity: None,
            density: None,
        };
        let params = ConservationParams {
            initial_energy: init,
            dt: 1e-6,
            temperature: 310.0,
            tolerance: 1e-4,
        };
        let m = validate_conservation(state, prev, params, &grid);

        assert_eq!(
            m.mass_error, 0.0,
            "None density_previous → mass_error must be 0"
        );
        assert_eq!(
            m.momentum_error,
            (0.0, 0.0, 0.0),
            "None velocity_previous → momentum_error must be (0,0,0)"
        );
    }

    /// Lossless field with matching initial energy → is_conserved = true.
    ///
    /// energy_error→0, mass_error=0, momentum_error=(0,0,0), ds_dt=0 ≥ 0 → is_conserved.
    #[test]
    fn validate_conservation_conserved_for_lossless_matched_energy() {
        let grid = small_grid();
        let s = (grid.nx, grid.ny, grid.nz);
        let p0 = 500.0_f64;
        let rho0 = 1000.0_f64;
        let c0 = 1500.0_f64;
        let p = uniform(s, p0);
        let v = uniform(s, 0.0);
        let rho = uniform(s, rho0);
        let c = uniform(s, c0);
        let alpha = uniform(s, 0.0);
        let dv = grid.dx * grid.dy * grid.dz;
        let n = (s.0 * s.1 * s.2) as f64;
        let init = p0.powi(2) / (2.0 * rho0 * c0.powi(2)) * dv * n;

        let state = AcousticStateRefs {
            pressure: &p,
            velocity_x: &v,
            velocity_y: &v,
            velocity_z: &v,
            density: &rho,
            sound_speed: &c,
            absorption: &alpha,
        };
        let prev = PreviousFields {
            pressure: None,
            velocity: None,
            density: None,
        };
        let params = ConservationParams {
            initial_energy: init,
            dt: 1e-6,
            temperature: 310.0,
            tolerance: 1e-4,
        };
        let m = validate_conservation(state, prev, params, &grid);

        assert!(
            m.energy_error < 1e-10,
            "energy_error={:.3e}",
            m.energy_error
        );
        assert!(
            m.entropy_production_rate >= 0.0,
            "entropy_production_rate must be nonneg"
        );
        assert!(
            m.is_conserved,
            "lossless matched field must satisfy is_conserved"
        );
    }
}
