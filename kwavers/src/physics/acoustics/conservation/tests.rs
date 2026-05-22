use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_K;
use crate::domain::grid::Grid;
use ndarray::Array3;

fn make_grid() -> Grid {
    Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap()
}

fn uniform_array(shape: (usize, usize, usize), val: f64) -> Array3<f64> {
    Array3::from_elem(shape, val)
}

#[test]
fn test_zero_energy_field_has_zero_relative_error() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p = uniform_array(shape, 0.0);
    let v = uniform_array(shape, 0.0);
    let rho = uniform_array(shape, 1000.0);
    let c = uniform_array(shape, SOUND_SPEED_WATER_SIM);
    let err = validate_energy_conservation(&p, &v, &v, &v, &rho, &c, 0.0, &grid);
    assert!(err < 1e-8, "Zero field energy error: {err:.3e}");
}

#[test]
fn test_energy_analytical_uniform_pressure() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p0 = 1000.0_f64;
    let rho0 = 1000.0_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let p = uniform_array(shape, p0);
    let v = uniform_array(shape, 0.0);
    let rho = uniform_array(shape, rho0);
    let c = uniform_array(shape, c0);
    let volume =
        (grid.nx as f64 * grid.dx) * (grid.ny as f64 * grid.dy) * (grid.nz as f64 * grid.dz);
    let expected = p0 * p0 / (2.0 * rho0 * c0 * c0) * volume;
    let err = validate_energy_conservation(&p, &v, &v, &v, &rho, &c, expected, &grid);
    assert!(err < 1e-10, "Energy relative error: {err:.3e}");
}

#[test]
fn test_entropy_production_zero_absorption() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p = uniform_array(shape, 1000.0);
    let v = uniform_array(shape, 0.0);
    let rho = uniform_array(shape, 1000.0);
    let c = uniform_array(shape, SOUND_SPEED_WATER_SIM);
    let alpha = uniform_array(shape, 0.0);
    let ds = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha, BODY_TEMPERATURE_K, &grid);
    assert!(ds.abs() < 1e-20, "Lossless entropy production: {ds:.3e}");
}

#[test]
fn test_entropy_production_non_negative() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p = uniform_array(shape, 5000.0);
    let v = uniform_array(shape, 0.1);
    let rho = uniform_array(shape, 1000.0);
    let c = uniform_array(shape, SOUND_SPEED_WATER_SIM);
    let alpha = uniform_array(shape, 2.0);
    let ds = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha, BODY_TEMPERATURE_K, &grid);
    assert!(
        ds >= 0.0,
        "Entropy production must be nonnegative: {ds:.3e}"
    );
}

#[test]
fn test_entropy_production_scales_linearly_with_absorption() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p = uniform_array(shape, 2000.0);
    let v = uniform_array(shape, 0.05);
    let rho = uniform_array(shape, 1000.0);
    let c = uniform_array(shape, SOUND_SPEED_WATER_SIM);
    let alpha1 = uniform_array(shape, 1.0);
    let alpha2 = uniform_array(shape, 2.0);
    let ds1 = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha1, 300.0, &grid);
    let ds2 = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha2, 300.0, &grid);
    let ratio = ds2 / ds1;
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "absorption scaling ratio={ratio}"
    );
}

#[test]
fn test_entropy_production_analytical() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p0 = 2000.0_f64;
    let rho0 = 1000.0_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let alpha0 = 3.0_f64;
    let t0 = 310.0_f64;
    let p = uniform_array(shape, p0);
    let v = uniform_array(shape, 0.0);
    let rho = uniform_array(shape, rho0);
    let c = uniform_array(shape, c0);
    let alpha = uniform_array(shape, alpha0);
    let volume =
        (grid.nx as f64 * grid.dx) * (grid.ny as f64 * grid.dy) * (grid.nz as f64 * grid.dz);
    let expected = alpha0 * p0 * p0 / (rho0 * c0 * t0) * volume;
    let ds = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha, t0, &grid);
    let rel = (ds - expected).abs() / expected;
    assert!(rel < 1e-10, "entropy rel={rel:.3e}");
}

#[test]
fn test_acoustic_intensity_pointwise() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p = uniform_array(shape, 3.0);
    let vx = uniform_array(shape, 2.0);
    let vy = uniform_array(shape, 0.0);
    let vz = uniform_array(shape, -1.0);
    let (ix, iy, iz) = acoustic_intensity(&p, &vx, &vy, &vz);
    assert!(ix.iter().all(|v| (v - 6.0).abs() < 1e-12));
    assert!(iy.iter().all(|v| v.abs() < 1e-12));
    assert!(iz.iter().all(|v| (v + 3.0).abs() < 1e-12));
}

#[test]
fn test_heat_source_zero_absorption() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p = uniform_array(shape, 5000.0);
    let v = uniform_array(shape, 1.0);
    let rho = uniform_array(shape, 1000.0);
    let c = uniform_array(shape, SOUND_SPEED_WATER_SIM);
    let alpha = uniform_array(shape, 0.0);
    let q = acoustic_heat_source(&p, &v, &v, &v, &rho, &c, &alpha);
    assert!(q.iter().all(|v| v.abs() < 1e-20));
}

#[test]
fn test_second_law_violation_detected() {
    let grid = make_grid();
    let shape = (grid.nx, grid.ny, grid.nz);
    let p = uniform_array(shape, 1000.0);
    let v = uniform_array(shape, 0.0);
    let rho = uniform_array(shape, 1000.0);
    let c = uniform_array(shape, SOUND_SPEED_WATER_SIM);
    let alpha = uniform_array(shape, -1.0);
    let volume =
        (grid.nx as f64 * grid.dx) * (grid.ny as f64 * grid.dy) * (grid.nz as f64 * grid.dz);
    let e_init = 1000.0 * 1000.0 / (2.0 * 1000.0 * SOUND_SPEED_WATER_SIM * SOUND_SPEED_WATER_SIM) * volume;
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
        initial_energy: e_init,
        dt: 1e-6,
        temperature: BODY_TEMPERATURE_K,
        tolerance: 1.0,
    };
    let metrics = validate_conservation(state, prev, params, &grid);
    assert!(metrics.entropy_production_rate < 0.0);
    assert!(!metrics.is_conserved);
}
