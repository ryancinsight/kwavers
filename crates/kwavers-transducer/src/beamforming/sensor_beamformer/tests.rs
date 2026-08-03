use super::beamformer::SensorBeamformer;
use super::types::{BeamformerWindowType, SensorProcessingParams};
use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond, Radian};
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_receiver::array::{Position, Sensor, SensorArray, SensorArrayGeometry};
use leto::Array2;

fn create_test_array(n_sensors: usize) -> SensorArray {
    let sensors: Vec<Sensor> = (0..n_sensors)
        .map(|i| {
            let position = Position {
                x: i as f64 * 0.001,
                y: 0.0,
                z: 0.0,
            };
            Sensor::new(i, position)
        })
        .collect();
    SensorArray::new(sensors, SOUND_SPEED_TISSUE, SensorArrayGeometry::Linear)
}

#[test]
fn test_windowing_preserves_dimensions() {
    let array = create_test_array(8);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));
    let delays = Array2::ones((8, 100));

    for window_type in [
        BeamformerWindowType::Hanning,
        BeamformerWindowType::Hamming,
        BeamformerWindowType::Blackman,
        BeamformerWindowType::Rectangular,
    ] {
        let windowed = beamformer.apply_windowing(&delays, window_type).unwrap();
        assert_eq!(windowed.shape(), delays.shape());
    }
}

#[test]
fn test_rectangular_window_is_identity() {
    let array = create_test_array(8);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));
    let delays = Array2::from_shape_fn((8, 100), |[i, j]| i as f64 + j as f64 * 0.1);

    let windowed = beamformer
        .apply_windowing(&delays, BeamformerWindowType::Rectangular)
        .unwrap();

    for i in 0..8 {
        for j in 0..100 {
            assert_relative_eq!(windowed[[i, j]], delays[[i, j]], epsilon = 1e-12);
        }
    }
}

#[test]
fn test_window_reduces_edge_elements() {
    let array = create_test_array(16);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));
    let delays = Array2::ones((16, 50));

    for window_type in [
        BeamformerWindowType::Hanning,
        BeamformerWindowType::Hamming,
        BeamformerWindowType::Blackman,
    ] {
        let windowed = beamformer.apply_windowing(&delays, window_type).unwrap();

        // Symmetric window: first and last element weights are equal.
        assert_relative_eq!(windowed[[0, 0]], windowed[[15, 0]], epsilon = 1e-10);

        let center_val = windowed[[8, 0]];
        let edge_val = windowed[[0, 0]];
        assert!(
            edge_val < center_val,
            "Edge value {edge_val} should be less than center {center_val} for {window_type:?}"
        );
    }
}

#[test]
fn test_hanning_window_has_zero_endpoints() {
    let array = create_test_array(8);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));
    let delays = Array2::ones((8, 10));
    let windowed = beamformer
        .apply_windowing(&delays, BeamformerWindowType::Hanning)
        .unwrap();

    assert!(windowed[[0, 0]].abs() < 1e-10);
    assert!(windowed[[7, 0]].abs() < 1e-10);
}

#[test]
fn test_windowing_applied_per_column() {
    let array = create_test_array(4);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));

    let mut delays = Array2::zeros((4, 3));
    delays
        .index_axis_mut::<1>(1, 0)
        .expect("column 0 in bounds")
        .fill(1.0);
    delays
        .index_axis_mut::<1>(1, 1)
        .expect("column 1 in bounds")
        .fill(2.0);
    delays
        .index_axis_mut::<1>(1, 2)
        .expect("column 2 in bounds")
        .fill(3.0);

    let windowed = beamformer
        .apply_windowing(&delays, BeamformerWindowType::Hamming)
        .unwrap();

    // All columns are scaled by the same per-row window coefficient,
    // so the ratio windowed[row, col] / delays[row, col] is identical
    // across columns for a given row.
    for col_idx in 1..3 {
        for row_idx in 0..4 {
            let ratio = windowed[[row_idx, col_idx]] / delays[[row_idx, col_idx]];
            let prev_ratio = windowed[[row_idx, col_idx - 1]] / delays[[row_idx, col_idx - 1]];
            assert_relative_eq!(ratio, prev_ratio, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_blackman_window_has_better_sidelobe_suppression() {
    let array = create_test_array(32);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));
    let delays = Array2::ones((32, 1));

    let hanning = beamformer
        .apply_windowing(&delays, BeamformerWindowType::Hanning)
        .unwrap();
    let blackman = beamformer
        .apply_windowing(&delays, BeamformerWindowType::Blackman)
        .unwrap();

    // Blackman tapers more aggressively; near-edge coefficient is lower.
    let edge_idx = 1;
    assert!(
        blackman[[edge_idx, 0]] < hanning[[edge_idx, 0]],
        "Blackman should suppress edge [{edge_idx}] more than Hanning"
    );
}

#[test]
fn test_windowing_with_zero_delays() {
    let array = create_test_array(8);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));
    let delays = Array2::zeros((8, 10));
    let windowed = beamformer
        .apply_windowing(&delays, BeamformerWindowType::Hanning)
        .unwrap();

    for i in 0..8 {
        for j in 0..10 {
            assert_eq!(windowed[[i, j]], 0.0);
        }
    }
}

#[test]
fn test_processing_params_f_number() {
    let params = SensorProcessingParams {
        n_sensors: 64,
        sampling_frequency: Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
        element_spacing: Length::from_unit::<Meter>(0.3e-3),
        array_aperture: Length::from_unit::<Meter>(19.2e-3),
    };

    let focal_length = Length::from_unit::<Meter>(50e-3);
    let f_num = params.f_number(focal_length).unwrap().into_base();
    assert_relative_eq!(f_num, 50e-3 / 19.2e-3, epsilon = 1e-10);
}

#[test]
fn test_processing_params_max_spatial_frequency() {
    let params = SensorProcessingParams {
        n_sensors: 64,
        sampling_frequency: Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
        element_spacing: Length::from_unit::<Meter>(0.3e-3),
        array_aperture: Length::from_unit::<Meter>(19.2e-3),
    };

    let sound_speed = Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE);
    let max_freq = params
        .max_spatial_frequency(sound_speed)
        .unwrap()
        .in_unit::<Hertz>();
    let expected = SOUND_SPEED_TISSUE / (2.0 * 0.3e-3);
    assert_relative_eq!(max_freq, expected, epsilon = 1e-6);
}

#[test]
fn test_calculate_delays_logic() {
    let sensors = vec![
        Sensor::new(
            0,
            Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        ),
        Sensor::new(
            1,
            Position {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        ),
    ];
    let array = SensorArray::new(sensors, SOUND_SPEED_TISSUE, SensorArrayGeometry::Linear);
    let beamformer = SensorBeamformer::new(array, Frequency::from_unit::<Hertz>(MHZ_TO_HZ));

    let grid = kwavers_grid::Grid::new(2, 1, 1, 1.0, 1.0, 1.0).unwrap();
    let delays = beamformer
        .calculate_delays(&grid, Velocity::from_unit::<MeterPerSecond>(1.0))
        .unwrap();

    // Sensor 0 at origin: d[0,0]=0, d[0,1]=1
    // Sensor 1 at x=1:    d[1,0]=1, d[1,1]=0
    assert_relative_eq!(delays[[0, 0]], 0.0);
    assert_relative_eq!(delays[[0, 1]], 1.0);
    assert_relative_eq!(delays[[1, 0]], 1.0);
    assert_relative_eq!(delays[[1, 1]], 0.0);
}

#[test]
fn processing_params_preserve_typed_geometry() {
    let beamformer = SensorBeamformer::new(
        create_test_array(3),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    );
    let params = beamformer.processing_params();

    assert_eq!(params.n_sensors, 3);
    assert_eq!(params.sampling_frequency.in_unit::<Hertz>(), MHZ_TO_HZ);
    assert_eq!(params.element_spacing.in_unit::<Meter>(), 1.0e-3);
    assert_eq!(params.array_aperture.in_unit::<Meter>(), 2.0e-3);
}

#[test]
fn positions_and_steering_use_typed_boundaries() {
    let beamformer = SensorBeamformer::new(
        create_test_array(2),
        Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
    );
    assert_eq!(beamformer.positions()[1][0].in_unit::<Meter>(), 1.0e-3);

    let result = beamformer
        .calculate_steering(
            &[
                (
                    Angle::from_unit::<Radian>(0.0),
                    Angle::from_unit::<Radian>(0.0),
                ),
                (
                    Angle::from_unit::<Radian>(std::f64::consts::FRAC_PI_2),
                    Angle::from_unit::<Radian>(0.0),
                ),
            ],
            Frequency::from_unit::<Hertz>(1_000.0),
            Velocity::from_unit::<MeterPerSecond>(1_500.0),
        )
        .unwrap();

    assert_eq!(result.shape(), [2, 2]);
    assert!((result[[0, 0]].norm() - 1.0).abs() < 1.0e-12);
}

#[test]
fn derived_metrics_reject_invalid_typed_inputs() {
    let params = SensorProcessingParams {
        n_sensors: 1,
        sampling_frequency: Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
        element_spacing: Length::from_base(0.0),
        array_aperture: Length::from_base(0.0),
    };

    assert!(params
        .f_number(Length::from_unit::<Meter>(50.0e-3))
        .is_err());
    assert!(params
        .max_spatial_frequency(Velocity::from_unit::<MeterPerSecond>(1_500.0))
        .is_err());
}
