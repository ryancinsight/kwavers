use super::GpuPstdSimulationAdapter;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER};
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;
use kwavers_domain::medium::homogeneous::HomogeneousMedium;
use kwavers_solver::config::{SolverConfiguration, SolverType};

#[test]
fn rejects_non_power_of_two_grid() {
    let grid = Grid::new(5, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let err = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap_err();

    assert!(matches!(err, KwaversError::InvalidInput(_)));
}

#[test]
fn rejects_axis_exceeding_256() {
    let grid = Grid::new(512, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let err = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap_err();

    assert!(matches!(err, KwaversError::InvalidInput(_)));
}

#[test]
fn constructs_for_valid_power_of_two_grid() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

    assert_eq!(adapter.name(), "GpuPstd");
    assert_eq!(adapter.pressure_field().dim(), (8, 8, 8));
    assert!(adapter.recorded_sensor_pressure().is_none());
}

#[test]
fn step_forward_returns_feature_not_available() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

    let err = adapter.step_forward().unwrap_err();
    assert!(matches!(err, KwaversError::FeatureNotAvailable(_)));
}

#[test]
fn add_sensor_rejects_out_of_bounds_point() {
    use kwavers_domain::sensor::grid_sampling::{GridPoint, GridSensorSet};

    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
    let sensor = GridSensorSet::from_points(vec![GridPoint::new(99, 0, 0)]);

    let err = adapter.add_sensor(&sensor).unwrap_err();
    assert!(matches!(err, KwaversError::InvalidInput(_)));
}

#[test]
fn add_sensor_valid_points_sets_mask() {
    use kwavers_domain::sensor::grid_sampling::{GridPoint, GridSensorSet};

    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
    let sensor = GridSensorSet::from_points(vec![GridPoint::new(1, 2, 3), GridPoint::new(4, 5, 6)]);
    adapter.add_sensor(&sensor).unwrap();

    // Mask contains exactly 2 `true` entries at the specified coordinates.
    let true_count = adapter.sensor_mask.iter().filter(|&&v| v).count();
    assert_eq!(true_count, 2);
    assert!(adapter.sensor_mask[[1, 2, 3]]);
    assert!(adapter.sensor_mask[[4, 5, 6]]);
}
