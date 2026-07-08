use super::processor::DelayAndSumPAM;
use super::types::{ApodizationType, DelayAndSumConfig, PamImagingMode};
use approx::assert_relative_eq;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array1;
use ndarray::Array2;

#[test]
fn test_pam_creation() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig::default();
    let _pam = DelayAndSumPAM::new(sensors, config).unwrap();
}

#[test]
fn test_insufficient_sensors() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]];
    let config = DelayAndSumConfig::default();
    assert!(DelayAndSumPAM::new(sensors, config).is_err());
}

#[test]
fn test_delay_computation() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig::default();
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let source_pos = [0.0, 0.0, 0.0];
    let delays = pam.compute_delays(&source_pos).unwrap();

    assert_eq!(delays.len(), 3);
    assert_relative_eq!(delays[0], 0.0, epsilon = 1e-6);
}

#[test]
fn test_apodization_weights() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig {
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let weights = pam.compute_apodization_weights();
    assert_eq!(weights.len(), 3);
    assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-6));
}

#[test]
fn test_beamform_basic() {
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.01, 0.01, 0.0],
    ];
    let config = DelayAndSumConfig::default();
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let passive_data = Array2::<f64>::from_shape_fn((4, 1000), |(i, t)| {
        (TWO_PI * t as f64 / 100.0 + i as f64).sin()
    });

    let grid_points = Array2::<f64>::from_shape_fn((5, 3), |(i, j)| match j {
        0 => (i as f64 - 2.0) * 0.005,
        1 => (i as f64 - 2.0) * 0.005,
        2 => 0.02,
        _ => 0.0,
    });

    let intensity_map = pam.beamform(&passive_data, &grid_points).unwrap();

    assert_eq!(intensity_map.len(), 5);
    assert!(intensity_map.iter().all(|&x| x >= 0.0));
}

#[test]
fn beamform_view_localizes_analytic_impulse_source() {
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let sampling_frequency = MHZ_TO_HZ;
    let sensors = vec![
        [-0.006, 0.0, 0.0],
        [-0.002, 0.0, 0.0],
        [0.002, 0.0, 0.0],
        [0.006, 0.0, 0.0],
    ];
    let source = [0.001, 0.0, 0.018];
    let distractor = [0.0, 0.0, 0.024];
    let config = DelayAndSumConfig {
        sound_speed,
        sampling_frequency,
        window_size: 64,
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors.clone(), config).unwrap();
    let mut passive_data = Array2::<f64>::zeros((sensors.len(), 96));

    for (sensor_idx, sensor) in sensors.iter().enumerate() {
        let dx = source[0] - sensor[0];
        let dy = source[1] - sensor[1];
        let dz = source[2] - sensor[2];
        let sample = ((dx * dx + dy * dy + dz * dz).sqrt() / sound_speed * sampling_frequency)
            .round() as usize;
        passive_data[[sensor_idx, sample]] = 1.0;
    }

    let grid_points = Array2::from_shape_vec(
        (2, 3),
        vec![
            source[0],
            source[1],
            source[2],
            distractor[0],
            distractor[1],
            distractor[2],
        ],
    )
    .unwrap();
    let intensity = pam
        .beamform_view(passive_data.view(), grid_points.view())
        .unwrap();

    assert_eq!(intensity.len(), 2);
    assert!(intensity[0].is_finite());
    assert!(intensity[1].is_finite());
    assert!(
        intensity[0] > 4.0 * intensity[1],
        "true-source intensity {} did not dominate distractor {}",
        intensity[0],
        intensity[1]
    );
}

#[test]
fn dmas_sharpens_localization_relative_to_das() {
    // A broadband (impulsive) point source recorded on an 8-element aperture.
    // DMAS replaces the coherent sum with the pairwise signal correlation, so
    // its source-to-sidelobe contrast must exceed delay-and-sum's
    // (Matrone et al. 2015).
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let sampling_frequency = 2.0 * MHZ_TO_HZ;
    let sensors: Vec<[f64; 3]> = (0..8)
        .map(|i| [(i as f64 - 3.5) * 0.004, 0.0, 0.0])
        .collect();
    let source = [0.0, 0.0, 0.030];

    let config = DelayAndSumConfig {
        sound_speed,
        sampling_frequency,
        window_size: 128,
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors.clone(), config).unwrap();

    // Impulse arrival per sensor at the true-source travel time.
    let n_samples = 256;
    let mut passive_data = Array2::<f64>::zeros((sensors.len(), n_samples));
    for (sensor_idx, sensor) in sensors.iter().enumerate() {
        let dx = source[0] - sensor[0];
        let dy = source[1] - sensor[1];
        let dz = source[2] - sensor[2];
        let sample = ((dx * dx + dy * dy + dz * dz).sqrt() / sound_speed * sampling_frequency)
            .round() as usize;
        passive_data[[sensor_idx, sample]] = 1.0;
    }

    // Axial scan line through the source; off-source pixels are sidelobes.
    let n_pixels = 41;
    let grid_points = Array2::from_shape_fn((n_pixels, 3), |(i, j)| match j {
        2 => 0.030 + (i as f64 - 20.0) * 0.0006,
        _ => 0.0,
    });
    let source_pixel = 20;

    let das = pam
        .beamform_with_mode_view(
            passive_data.view(),
            grid_points.view(),
            PamImagingMode::DelayAndSum,
        )
        .unwrap();
    let dmas = pam
        .beamform_with_mode_view(
            passive_data.view(),
            grid_points.view(),
            PamImagingMode::DelayMultiplyAndSum,
        )
        .unwrap();

    // Source-pixel value is positive for both imaging conditions.
    assert!(
        das[source_pixel] > 0.0,
        "DAS source intensity must be positive"
    );
    assert!(
        dmas[source_pixel] > 0.0,
        "DMAS source intensity must be positive"
    );

    // Source-to-max-sidelobe contrast (sidelobes = pixels >2 bins from source).
    let max_sidelobe = |map: &Array1<f64>| -> f64 {
        map.iter()
            .enumerate()
            .filter(|(i, _)| i.abs_diff(source_pixel) > 2)
            .map(|(_, &v)| v)
            .fold(0.0_f64, f64::max)
    };
    let das_contrast = das[source_pixel] / max_sidelobe(&das).max(f64::MIN_POSITIVE);
    let dmas_contrast = dmas[source_pixel] / max_sidelobe(&dmas).max(f64::MIN_POSITIVE);

    assert!(
        dmas_contrast > das_contrast,
        "DMAS must improve source-to-sidelobe contrast over DAS: dmas={dmas_contrast:.4e} das={das_contrast:.4e}"
    );
}

#[test]
fn beamform_with_delays_aligns_on_supplied_delays() {
    // Three sensors, an impulse in each at distinct samples. A delay row equal
    // to those arrival samples coherently re-aligns the impulses to t = 0
    // (energy ≈ N²); a zero-delay row leaves them outside the window (≈ 0).
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig {
        sound_speed: 1.0,
        sampling_frequency: 1.0,
        window_size: 8,
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let nt = 16;
    let arrivals = [10usize, 12, 14];
    let mut data = Array2::<f64>::zeros((3, nt));
    for (sensor, &arrival) in arrivals.iter().enumerate() {
        data[[sensor, arrival]] = 1.0;
    }

    let delays = Array2::from_shape_vec(
        (2, 3),
        vec![
            10.0, 12.0, 14.0, /* aligned */ 0.0, 0.0, 0.0, /* mis-aligned */
        ],
    )
    .unwrap();
    let signals = pam
        .beamform_signals_with_delays(data.view(), delays.view())
        .unwrap();

    let energy_aligned: f64 = signals.row(0).iter().map(|&x| x * x).sum();
    let energy_misaligned: f64 = signals.row(1).iter().map(|&x| x * x).sum();
    assert!(
        energy_aligned > 8.0,
        "aligned delays must coherently sum impulses (energy ≈ 9): {energy_aligned}"
    );
    assert!(
        energy_misaligned < 1e-9,
        "mis-aligned delays must leave the impulses outside the window: {energy_misaligned}"
    );
}

#[test]
fn beamform_signals_with_delays_rejects_bad_shapes() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let pam = DelayAndSumPAM::new(sensors, DelayAndSumConfig::default()).unwrap();
    let data = Array2::<f64>::zeros((3, 16));
    // Delay matrix with the wrong number of sensor columns.
    let bad = Array2::<f64>::zeros((4, 2));
    assert!(pam
        .beamform_signals_with_delays(data.view(), bad.view())
        .is_err());
    // Negative delay.
    let mut neg = Array2::<f64>::zeros((1, 3));
    neg[[0, 0]] = -1.0;
    assert!(pam
        .beamform_signals_with_delays(data.view(), neg.view())
        .is_err());
}

#[test]
fn beamform_view_uses_fractional_delay_interpolation() {
    let sensors = vec![[0.0, 0.0, 0.0]; 3];
    let config = DelayAndSumConfig {
        sound_speed: 1.0,
        sampling_frequency: 1.0,
        window_size: 1,
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();
    let passive_data = Array2::from_shape_vec((3, 2), vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0]).unwrap();
    let grid_points = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.5]).unwrap();

    let intensity = pam
        .beamform_view(passive_data.view(), grid_points.view())
        .unwrap();

    assert_relative_eq!(intensity[0], 9.0, epsilon = 1e-12);
}

#[test]
fn beamform_view_rejects_invalid_boundary_shapes_and_values() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let pam = DelayAndSumPAM::new(sensors, DelayAndSumConfig::default()).unwrap();
    let passive_data = Array2::<f64>::zeros((3, 16));
    let bad_grid_shape = Array2::<f64>::zeros((2, 2));
    assert!(pam
        .beamform_view(passive_data.view(), bad_grid_shape.view())
        .is_err());

    let mut nonfinite_data = passive_data.clone();
    nonfinite_data[[0, 0]] = f64::NAN;
    let grid_points = Array2::<f64>::zeros((1, 3));
    assert!(pam
        .beamform_view(nonfinite_data.view(), grid_points.view())
        .is_err());
}

#[test]
fn test_event_detection() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig {
        detection_threshold: 2.0,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let intensity_map =
        Array1::from_vec([5], vec![0.5, 0.8, 5.0, 1.0, 0.3]).expect("shape matches map length");
    let grid_points = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.0, 0.0, 0.02, 0.01, 0.0, 0.02, 0.005, 0.005, 0.02, 0.0, 0.01, 0.02, -0.01, 0.0, 0.02,
        ],
    )
    .unwrap();

    let events = pam
        .detect_events(&intensity_map, &grid_points, 0.0)
        .unwrap();

    assert!(!events.is_empty());
    assert!(events[0].intensity > 2.0);
}

#[test]
fn test_event_detection_with_peak_frequency() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let config = DelayAndSumConfig {
        sampling_frequency: 10.0 * MHZ_TO_HZ,
        window_size: 256,
        detection_threshold: 0.5,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let freq = MHZ_TO_HZ;
    let num_samples = 256;
    let mut passive_data = Array2::zeros((3, num_samples));
    for t in 0..num_samples {
        let time = t as f64 / pam.config.sampling_frequency;
        let sample = (TWO_PI * freq * time).sin();
        for sensor in 0..3 {
            passive_data[[sensor, t]] = sample;
        }
    }

    let intensity_map = Array1::from_vec([1], vec![2.0]).expect("shape matches map length");
    let grid_points = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();

    let events = pam
        .detect_events_with_data(&passive_data, &intensity_map, &grid_points, 0.0)
        .unwrap();

    assert!(!events.is_empty());
    let peak = events[0]
        .peak_frequency
        .expect("peak frequency should be available");
    let resolution = pam.config.sampling_frequency / num_samples as f64;
    assert!((peak - freq).abs() <= resolution);
}
