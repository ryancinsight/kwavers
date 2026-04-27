//! Aggregated conservation validation.

use super::{
    entropy_production_rate, validate_energy_conservation, validate_mass_conservation,
    validate_momentum_conservation, ConservationMetrics,
};
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Run all conservation checks and return consolidated metrics.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn validate_conservation(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
    temperature: f64,
    _pressure_previous: Option<&Array3<f64>>,
    velocity_x_previous: Option<&Array3<f64>>,
    velocity_y_previous: Option<&Array3<f64>>,
    velocity_z_previous: Option<&Array3<f64>>,
    density_previous: Option<&Array3<f64>>,
    initial_energy: f64,
    dt: f64,
    grid: &Grid,
    tolerance: f64,
) -> ConservationMetrics {
    let energy_error = validate_energy_conservation(
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
        density,
        sound_speed,
        initial_energy,
        grid,
    );
    let mass_error = if let Some(rho_prev) = density_previous {
        validate_mass_conservation(
            density, rho_prev, velocity_x, velocity_y, velocity_z, dt, grid,
        )
    } else {
        0.0
    };
    let momentum_error = if let (Some(vx_prev), Some(vy_prev), Some(vz_prev)) = (
        velocity_x_previous,
        velocity_y_previous,
        velocity_z_previous,
    ) {
        validate_momentum_conservation(
            velocity_x, velocity_y, velocity_z, vx_prev, vy_prev, vz_prev, pressure, density, dt,
            grid,
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    let ds_dt = entropy_production_rate(
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
        density,
        sound_speed,
        absorption,
        temperature,
        grid,
    );
    let is_conserved = energy_error < tolerance
        && mass_error < tolerance
        && momentum_error.0 < tolerance
        && momentum_error.1 < tolerance
        && momentum_error.2 < tolerance
        && ds_dt >= 0.0;

    ConservationMetrics {
        energy_error,
        mass_error,
        momentum_error,
        entropy_production_rate: ds_dt,
        is_conserved,
    }
}
