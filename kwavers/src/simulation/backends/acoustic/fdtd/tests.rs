//! Tests for `FdtdBackend`.

use super::super::backend::AcousticSolverBackend;
use super::backend::FdtdBackend;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::physics::acoustics::mechanics::acoustic_wave::AcousticSpatialOrder;

fn create_test_grid() -> Grid {
    Grid::new(32, 32, 32, 0.0005, 0.0005, 0.0005).expect("Failed to create test grid")
}

fn create_test_medium(grid: &Grid) -> HomogeneousMedium {
    HomogeneousMedium::new(1000.0, SOUND_SPEED_WATER_SIM, 0.0, 0.0, grid)
}

#[test]
fn test_fdtd_backend_creation() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);

    let backend = FdtdBackend::new(&grid, &medium, AcousticSpatialOrder::Second).unwrap();
    assert_eq!(backend.get_grid_dimensions(), (32, 32, 32));
    assert!(backend.get_dt() > 0.0, "Time step must be positive");
    assert_eq!(backend.get_current_time(), 0.0);
}

#[test]
fn test_fdtd_backend_cfl_condition() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);

    let backend = FdtdBackend::new(&grid, &medium, AcousticSpatialOrder::Second).unwrap();

    let dt = backend.get_dt();
    let dx = grid.min_spacing();
    let c_max = SOUND_SPEED_WATER_SIM;
    let cfl = c_max * dt / dx;
    let cfl_limit = 1.0 / 3.0_f64.sqrt();

    assert!(
        cfl < cfl_limit,
        "CFL condition violated: {} >= {}",
        cfl,
        cfl_limit
    );
    assert!(cfl < 0.6, "CFL factor should be conservative: {}", cfl);
}

#[test]
fn test_fdtd_backend_time_stepping() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);

    let mut backend = FdtdBackend::new(&grid, &medium, AcousticSpatialOrder::Second).unwrap();

    let dt = backend.get_dt();
    assert_eq!(backend.get_current_time(), 0.0);

    backend.step().expect("First step failed");
    assert!((backend.get_current_time() - dt).abs() < 1e-15);

    backend.step().expect("Second step failed");
    assert!((backend.get_current_time() - 2.0 * dt).abs() < 1e-14);
}

#[test]
fn test_fdtd_backend_field_access() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);
    let backend = FdtdBackend::new(&grid, &medium, AcousticSpatialOrder::Second).unwrap();

    let p = backend.get_pressure_field();
    assert_eq!(p.shape(), &[32, 32, 32]);

    let (vx, vy, vz) = backend.get_velocity_fields();
    assert_eq!(vx.shape(), &[32, 32, 32]);
    assert_eq!(vy.shape(), &[32, 32, 32]);
    assert_eq!(vz.shape(), &[32, 32, 32]);
}

#[test]
fn test_fdtd_backend_intensity_computation() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);
    let backend = FdtdBackend::new(&grid, &medium, AcousticSpatialOrder::Second).unwrap();

    let intensity = backend
        .get_intensity_field()
        .expect("Intensity computation failed");
    assert_eq!(intensity.shape(), &[32, 32, 32]);

    let max_intensity = intensity.iter().cloned().fold(0.0_f64, f64::max);
    assert_eq!(max_intensity, 0.0);
}

#[test]
fn test_fdtd_backend_as_trait_object() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);

    let backend = FdtdBackend::new(&grid, &medium, AcousticSpatialOrder::Second).unwrap();
    let mut solver: Box<dyn AcousticSolverBackend> = Box::new(backend);

    assert_eq!(solver.get_grid_dimensions(), (32, 32, 32));
    assert!(solver.get_dt() > 0.0);

    solver.step().expect("Step failed");
    assert!(solver.get_current_time() > 0.0);
}

#[test]
fn test_stable_timestep_computation() {
    let dx = 0.0005;
    let c = SOUND_SPEED_WATER_SIM;
    let dt = FdtdBackend::compute_stable_timestep(dx, c);

    let cfl = c * dt / dx;
    assert!(cfl < 1.0 / 3.0_f64.sqrt());
    assert!(dt > 1e-8 && dt < 1e-6, "Unexpected time step: {}", dt);
}

#[test]
fn test_max_sound_speed_estimation() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);

    let c_max = FdtdBackend::estimate_max_sound_speed(&medium, &grid).unwrap();
    assert!((c_max - SOUND_SPEED_WATER_SIM).abs() < 1e-6);
}

#[test]
fn test_spatial_order_variants() {
    let grid = create_test_grid();
    let medium = create_test_medium(&grid);

    for order in &[
        AcousticSpatialOrder::Second,
        AcousticSpatialOrder::Fourth,
        AcousticSpatialOrder::Sixth,
    ] {
        let backend = FdtdBackend::new(&grid, &medium, *order)
            .unwrap_or_else(|e| panic!("Failed to create backend with order {order:?}: {e:?}"));
        assert_eq!(
            backend.get_grid_dimensions(),
            (32, 32, 32),
            "order {order:?}: grid dimensions must match"
        );
        assert!(
            backend.get_dt() > 0.0,
            "order {order:?}: dt must be positive"
        );
    }
}
