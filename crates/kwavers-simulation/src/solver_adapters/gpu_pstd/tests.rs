use super::GpuPstdSimulationAdapter;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER};
use kwavers_core::error::KwaversError;
use kwavers_gpu::pstd_gpu::PstdFinalFields;
use kwavers_grid::Grid;
use kwavers_medium::{heterogeneous::HeterogeneousMedium, homogeneous::HomogeneousMedium};
use kwavers_solver::config::{FftBackend, SolverConfiguration, SolverType};
use kwavers_solver::Solver;
use kwavers_source::NullSource;

#[test]
fn accepts_non_power_of_two_grid() {
    let grid = Grid::new(5, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium)
        .expect("Hephaestus supports Bluestein transforms for non-power-of-two axes");

    assert_eq!(adapter.pressure_field().shape(), [5, 8, 8]);
}

#[test]
fn rejects_heterogeneous_absorption_exponents_during_factory_preparation() {
    let grid = Grid::new(2, 2, 2, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let mut medium = HeterogeneousMedium::new(2, 2, 2, false);
    medium
        .sound_speed
        .iter_mut()
        .for_each(|value| *value = 1_500.0);
    medium.density.iter_mut().for_each(|value| *value = 1_000.0);
    medium.absorption.iter_mut().for_each(|value| *value = 0.2);
    medium.alpha_power.iter_mut().for_each(|value| *value = 1.3);
    medium.alpha_power[[0, 0, 1]] = 1.7;
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let error = GpuPstdSimulationAdapter::new(&config, &grid, &medium)
        .expect_err("one GPU spectral symbol cannot represent multiple exponents");

    assert_eq!(
        error.to_string(),
        "Feature not available: Hephaestus PSTD requires one active absorption exponent; medium flat index 1 uses 1.7"
    );
}

#[test]
fn accepts_large_fft_axis_without_local_shader_limit() {
    let grid = Grid::new(1024, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium)
        .expect("FFT axis length is delegated to Hephaestus");

    assert_eq!(adapter.pressure_field().shape(), [1024, 8, 8]);
}

#[test]
fn accepts_axis_exceeding_retired_local_fft_limit() {
    let grid = Grid::new(2048, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium)
        .expect("the retired local shader limit must not constrain construction");

    assert_eq!(adapter.pressure_field().shape(), [2048, 8, 8]);
}

#[test]
fn constructs_for_valid_power_of_two_grid() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

    assert_eq!(adapter.name(), "GpuPstd");
    assert_eq!(adapter.pressure_field().shape(), [8, 8, 8]);
    assert!(adapter.recorded_sensor_pressure().is_none());
}

/// `len` consecutive whole numbers from `start`, each exact in `f32`.
fn ramp(start: f32, len: usize) -> Vec<f32> {
    std::iter::successors(Some(start), |value| Some(value + 1.0))
        .take(len)
        .collect()
}

/// Three distinct extents: a transfer that reorders the axes cannot put the
/// flat provider index back on the element it belongs to.
#[test]
fn final_field_readback_populates_solver_field_contract() {
    let grid = Grid::new(5, 6, 7, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };
    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
    let field_len = grid.nx * grid.ny * grid.nz;
    assert_eq!(field_len, 210);

    adapter
        .store_final_fields(PstdFinalFields {
            pressure: ramp(0.0, field_len),
            velocity_x: ramp(1000.0, field_len),
            velocity_y: ramp(2000.0, field_len),
            velocity_z: ramp(3000.0, field_len),
        })
        .expect("field vectors match the adapter grid");

    // Row-major: the provider's flat index is `(i * ny + j) * nz + k`.
    let (velocity_x, velocity_y, velocity_z) = adapter.velocity_fields();
    let mut flat = 0.0_f64;
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                assert_eq!(adapter.pressure_field()[[i, j, k]], flat);
                assert_eq!(velocity_x[[i, j, k]], 1000.0 + flat);
                assert_eq!(velocity_y[[i, j, k]], 2000.0 + flat);
                assert_eq!(velocity_z[[i, j, k]], 3000.0 + flat);
                flat += 1.0;
            }
        }
    }
    let statistics = adapter.statistics();
    assert_eq!(statistics.max_pressure, 209.0);
    assert_eq!(statistics.max_velocity, 3209.0);
}

/// The pressure record is the last pass of a time step (`encode_time_step`
/// in `kwavers-gpu`), so the last sample of a sensor trace and the final
/// pressure at that sensor are one provider value read back twice.
#[test]
#[ignore = "requires a GPU device"]
fn run_exposes_the_final_pressure_the_batch_recorded() {
    use kwavers_receiver::grid_sampling::{GridPoint, GridSensorSet};
    use kwavers_source::GridSource;
    use leto::{Array2, Array3};

    const STEPS: usize = 3;
    let grid = Grid::new(7, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };
    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
    let mut source_mask = Array3::zeros([7, 4, 3]);
    source_mask[[3, 2, 1]] = 1.0;
    adapter.set_grid_source(GridSource {
        p_mask: Some(source_mask),
        p_signal: Some(
            Array2::from_shape_vec([1, STEPS], vec![0.0, 1.0, 0.0])
                .expect("one source row with one sample per step"),
        ),
        ..GridSource::new_empty()
    });
    // In row-major order, as the provider numbers its sensors.
    let sensors = [[3, 2, 1], [5, 1, 2]];
    adapter
        .add_sensor(&GridSensorSet::from_points(
            sensors
                .iter()
                .map(|&[i, j, k]| GridPoint::new(i, j, k))
                .collect(),
        ))
        .unwrap();

    adapter.run(STEPS).expect("the device executes the batch");

    let recorded = adapter
        .recorded_sensor_pressure()
        .expect("the batch recorded both sensors");
    assert_eq!(recorded.shape(), [2, STEPS]);
    for (row, &[i, j, k]) in sensors.iter().enumerate() {
        let last_sample = recorded[[row, STEPS - 1]];
        assert_ne!(last_sample, 0.0, "sensor {row} recorded no pressure");
        assert_eq!(adapter.pressure_field()[[i, j, k]], last_sample);
    }
}

#[test]
fn peak_field_readback_preserves_the_provider_envelope_contract() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };
    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
    let field_len = grid.nx * grid.ny * grid.nz;
    let expected: Vec<f32> = (0..field_len).map(|index| index as f32 + 0.25).collect();

    adapter
        .store_peak_pressure(expected)
        .expect("peak-pressure values match the adapter grid and are non-negative");

    let peak = adapter
        .peak_pressure_field()
        .expect("explicit peak readback is retained separately from final pressure");
    assert_eq!(peak[[0, 0, 0]], 0.25);
    assert_eq!(peak[[7, 7, 7]], 511.25);
    assert_eq!(adapter.statistics().max_pressure, 511.25);
}

#[test]
fn peak_field_readback_rejects_negative_provider_values() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };
    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();
    let mut invalid = vec![0.0; grid.nx * grid.ny * grid.nz];
    invalid[7] = -1.0;

    let error = adapter.store_peak_pressure(invalid).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Invalid input: GPU PSTD peak-pressure readback has negative value -1 at flat index 7"
    );
    assert!(adapter.peak_pressure_field().is_none());
}

#[test]
fn step_forward_returns_feature_not_available() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };

    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

    let err = adapter.step_forward().unwrap_err();
    assert!(matches!(err, KwaversError::FeatureNotAvailable(_)));
}

#[test]
fn add_source_rejects_an_unsampled_waveform() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };
    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

    let error = adapter
        .add_source(Box::new(NullSource::new()))
        .expect_err("a Source trait object cannot encode GPU source-major samples");

    assert_eq!(
        error.to_string(),
        "Feature not available: GpuPstdSimulationAdapter requires a source-major sampled waveform; use set_grid_source"
    );
}

#[test]
fn peak_run_rejects_zero_steps_before_device_acquisition() {
    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
        dt: 1.0e-7,
        ..SolverConfiguration::default()
    };
    let mut adapter = GpuPstdSimulationAdapter::new(&config, &grid, &medium).unwrap();

    let error = adapter
        .run_peak_pressure(0)
        .expect_err("zero time steps violate the GPU PSTD batch contract");

    assert_eq!(
        error.to_string(),
        "Invalid input: GPU PSTD requires time_steps > 0 and finite positive dt; got steps=0 dt=0.0000001"
    );
}

#[test]
fn add_sensor_rejects_out_of_bounds_point() {
    use kwavers_receiver::grid_sampling::{GridPoint, GridSensorSet};

    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
    let config = SolverConfiguration {
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
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
        solver_type: SolverType::PSTD,
        fft_backend: FftBackend::Hephaestus,
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
