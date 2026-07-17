use super::GpuPstdSimulationAdapter;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER};
use kwavers_core::error::KwaversError;
use kwavers_gpu::pstd_gpu::PstdFinalFields;
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_solver::config::{SolverConfiguration, SolverType};
use kwavers_solver::Solver;

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
fn accepts_the_1024_point_fft_axis() {
    let grid = Grid::new(1024, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium)
        .expect("1024-point FFT axis is within the GPU PSTD contract");

    assert_eq!(adapter.pressure_field().shape(), [1024, 8, 8]);
}

#[test]
fn rejects_axis_exceeding_1024() {
    let grid = Grid::new(2048, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let err = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap_err();

    assert_eq!(
        err.to_string(),
        "Invalid input: GPU PSTD supports per-axis N ≤ 1024; got 2048×8×8"
    );
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
    assert_eq!(adapter.pressure_field().shape(), [8, 8, 8]);
    assert!(adapter.recorded_sensor_pressure().is_none());
}

#[test]
fn final_field_readback_populates_solver_field_contract() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PstdGpu,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };
    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
    let field_len = grid.nx * grid.ny * grid.nz;

    adapter
        .store_final_fields(PstdFinalFields {
            pressure: (0..field_len).map(|index| index as f32).collect(),
            velocity_x: (0..field_len).map(|index| (1000 + index) as f32).collect(),
            velocity_y: (0..field_len).map(|index| (2000 + index) as f32).collect(),
            velocity_z: (0..field_len).map(|index| (3000 + index) as f32).collect(),
        })
        .expect("field vectors match the adapter grid");

    assert_eq!(adapter.pressure_field()[[0, 0, 0]], 0.0);
    assert_eq!(adapter.pressure_field()[[7, 7, 7]], 511.0);
    let (velocity_x, velocity_y, velocity_z) = adapter.velocity_fields();
    assert_eq!(velocity_x[[7, 7, 7]], 1511.0);
    assert_eq!(velocity_y[[7, 7, 7]], 2511.0);
    assert_eq!(velocity_z[[7, 7, 7]], 3511.0);
    let statistics = adapter.statistics();
    assert_eq!(statistics.max_pressure, 511.0);
    assert_eq!(statistics.max_velocity, 3511.0);
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
    use kwavers_receiver::grid_sampling::{GridPoint, GridSensorSet};

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
    use kwavers_receiver::grid_sampling::{GridPoint, GridSensorSet};

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
