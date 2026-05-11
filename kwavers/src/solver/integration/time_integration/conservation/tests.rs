use super::*;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use ndarray::Array3;

#[test]
fn test_conservation_monitoring() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let mut monitor = ConservationMonitor::new(&grid);

    // Create initial conserved quantities
    let initial = ConservedQuantities {
        mass: 1000.0,
        momentum: (0.0, 0.0, 0.0),
        energy: 1e6,
        angular_momentum: (0.0, 0.0, 0.0),
    };

    monitor.set_initial(initial.clone());

    // Test conservation check with no change
    let error = monitor.check_conservation(0.1, initial.clone()).unwrap();
    assert!(error.max_error() < 1e-10);

    // Test conservation check with small change
    let mut changed = initial.clone();
    changed.mass *= 1.001; // 0.1% change
    let error = monitor.check_conservation(0.2, changed).unwrap();
    assert!(error.mass_error > 0.0);
    assert!(error.mass_error < 0.002);
}

#[test]
fn test_energy_computation() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let monitor = ConservationMonitor::new(&grid);
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);

    // Create test fields
    let pressure = Array3::from_elem((10, 10, 10), 1e5); // Pa
    let velocity_x = Array3::zeros((10, 10, 10));
    let velocity_y = Array3::zeros((10, 10, 10));
    let velocity_z = Array3::zeros((10, 10, 10));

    // Compute total energy
    let energy =
        monitor.compute_total_energy(&pressure, &velocity_x, &velocity_y, &velocity_z, &medium);

    // Energy should be positive
    assert!(energy > 0.0);

    // Compute acoustic energy
    let acoustic_energy = monitor.compute_acoustic_energy(&pressure, &medium);
    assert!(acoustic_energy > 0.0);
}
