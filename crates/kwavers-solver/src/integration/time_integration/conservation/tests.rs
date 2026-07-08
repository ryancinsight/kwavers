use super::*;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use ndarray::{Array3, ShapeBuilder};

fn assert_relative_roundoff_eq(actual: f64, expected: f64, operations: usize) {
    let tolerance = f64::EPSILON * operations as f64 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.17e}, got {actual:.17e}, tolerance {tolerance:.17e}"
    );
}

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
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);

    // Create test fields
    let pressure = Array3::from_elem((10, 10, 10), 1e5); // Pa
    let velocity_x = Array3::zeros((10, 10, 10));
    let velocity_y = Array3::zeros((10, 10, 10));
    let velocity_z = Array3::zeros((10, 10, 10));

    // Compute total energy
    let energy =
        monitor.compute_total_energy(&pressure, &velocity_x, &velocity_y, &velocity_z, &medium);

    let dv = grid.dx * grid.dy * grid.dz;
    let expected_energy = 1000.0 * 1e5 / 0.1 * dv;
    assert_relative_roundoff_eq(energy, expected_energy, pressure.len());

    // Compute acoustic energy
    let acoustic_energy = monitor.compute_acoustic_energy(&pressure, &medium);
    let expected_acoustic_energy =
        1000.0 * 1e10 / (2.0 * DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM.powi(2)) * dv;
    assert_relative_roundoff_eq(acoustic_energy, expected_acoustic_energy, pressure.len());
}

#[test]
fn energy_computation_preserves_logical_order_for_nonstandard_layouts() {
    let shape = (2, 3, 2);
    let grid = Grid::new(shape.0, shape.1, shape.2, 0.5, 0.25, 0.125).unwrap();
    let monitor = ConservationMonitor::new(&grid);
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);

    let pressure = Array3::from_shape_fn(shape.f(), |(i, j, k)| {
        10.0 * (1 + i) as f64 + 2.0 * j as f64 + k as f64
    });
    let velocity_x = Array3::from_shape_fn(shape, |(i, j, k)| 0.1 * (1 + i + j + k) as f64);
    let velocity_y = Array3::from_shape_fn(shape.f(), |(i, j, k)| 0.2 * (1 + 2 * i + j + k) as f64);
    let velocity_z = Array3::from_shape_fn(shape, |(i, j, k)| 0.3 * (1 + i + 2 * j + k) as f64);

    assert!(pressure.as_slice().is_none());
    assert!(velocity_y.as_slice().is_none());

    let mut expected_total = 0.0;
    let mut expected_acoustic = 0.0;
    let mut expected_potential = 0.0;
    let dv = grid.dx * grid.dy * grid.dz;
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let p = pressure[(i, j, k)];
                let vx = velocity_x[(i, j, k)];
                let vy = velocity_y[(i, j, k)];
                let vz = velocity_z[(i, j, k)];
                let kinetic = 0.5 * 1000.0 * vz.mul_add(vz, vx.mul_add(vx, vy * vy));
                expected_total += (kinetic + p / 0.1) * dv;

                let potential = p * p / (2.0 * 1000.0 * 1500.0_f64.powi(2));
                expected_acoustic += (potential + kinetic) * dv;
                expected_potential += potential * dv;
            }
        }
    }

    assert_eq!(
        monitor.compute_total_energy(&pressure, &velocity_x, &velocity_y, &velocity_z, &medium),
        expected_total
    );
    assert_eq!(
        monitor.compute_acoustic_energy_with_velocity(
            &pressure,
            Some(&velocity_x),
            Some(&velocity_y),
            Some(&velocity_z),
            &medium
        ),
        expected_acoustic
    );
    assert_eq!(
        monitor.compute_acoustic_energy(&pressure, &medium),
        expected_potential
    );
}
