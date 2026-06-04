use super::AcousticWaveSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;

fn create_test_grid() -> Grid {
    Grid::new(32, 32, 32, 0.0005, 0.0005, 0.0005).expect("Failed to create grid")
}

fn create_water_medium(grid: &Grid) -> HomogeneousMedium {
    HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL, // SSOT: fundamental::DENSITY_WATER_NOMINAL
        SOUND_SPEED_WATER_SIM, // Water sound speed
        0.0,                   // Optical absorption mu_a
        0.0,                   // Optical scattering mu_s_prime
        grid,
    )
}

#[test]
fn test_acoustic_solver_creation() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();
    assert_eq!(solver.grid_dimensions(), (32, 32, 32));
    assert!(solver.timestep() > 0.0);
    assert_eq!(solver.current_time(), 0.0);
}

#[test]
fn test_acoustic_solver_time_stepping() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    let dt = solver.timestep();

    solver.step().expect("Step failed");
    assert!((solver.current_time() - dt).abs() < 1e-15);

    solver.step().expect("Second step failed");
    assert!((solver.current_time() - 2.0 * dt).abs() < 1e-14);
}

#[test]
fn test_acoustic_solver_advance() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    let duration = 1e-6;
    solver.advance(duration).expect("Advance failed");

    assert!(solver.current_time() >= duration);
    assert!(solver.current_time() < duration + solver.timestep());
}

#[test]
fn test_acoustic_solver_field_access() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    let p = solver.pressure_field();
    assert_eq!(p.shape(), &[32, 32, 32]);

    let (vx, vy, vz) = solver.velocity_fields();
    assert_eq!(vx.shape(), &[32, 32, 32]);
    assert_eq!(vy.shape(), &[32, 32, 32]);
    assert_eq!(vz.shape(), &[32, 32, 32]);

    let intensity = solver.intensity_field().expect("Intensity failed");
    assert_eq!(intensity.shape(), &[32, 32, 32]);
}

#[test]
fn test_acoustic_solver_max_pressure() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    let p_max = solver.max_pressure();
    assert_eq!(p_max, 0.0);
}

#[test]
fn test_acoustic_solver_spta_intensity() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    let i_spta = solver.spta_intensity(1e-3).expect("SPTA failed");
    assert_eq!(i_spta, 0.0);
}

#[test]
fn test_advance_negative_duration() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    let result = solver.advance(-1e-6);
    assert!(result.is_err());
}

#[test]
fn test_advance_zero_duration() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);

    let mut solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    solver.advance(0.0).expect("Zero advance failed");
    assert_eq!(solver.current_time(), 0.0);
}

#[test]
fn test_spta_intensity_validation() {
    let grid = create_test_grid();
    let medium = create_water_medium(&grid);
    let solver = AcousticWaveSolver::new(&grid, &medium).unwrap();

    solver.spta_intensity(-1.0).unwrap_err();
    solver.spta_intensity(0.0).unwrap_err();

    assert_eq!(
        solver.spta_intensity(1.0).unwrap(),
        0.0,
        "zero-field SPTA intensity must be 0.0"
    );
}
