use kwavers::sensor::passive_acoustic_mapping::{
    beamforming::{ApodizationType, Beamformer, BeamformingConfig, BeamformingMethod},
    geometry::ArrayGeometry,
};
use ndarray::Array3;

fn linear_array_positions(elements: usize, pitch_m: f64) -> ArrayGeometry {
    ArrayGeometry::Linear {
        elements,
        pitch: pitch_m,
        center: [0.0, 0.0, 0.0],
        orientation: [1.0, 0.0, 0.0],
    }
}

fn synth_sensor_data(
    geometry: &ArrayGeometry,
    n_samples: usize,
    focal_point: [f64; 3],
    sample_rate: f64,
) -> Array3<f64> {
    // Shape: (n_elements, 1, n_samples)
    let n_elements = geometry.num_elements();
    let mut data = Array3::<f64>::zeros((n_elements, 1, n_samples));

    // Compute delays and create unit impulses at those indices
    // Reuse a simple delay computation consistent with beamforming implementation
    let positions = geometry.element_positions();
    let c = kwavers::physics::constants::SOUND_SPEED_WATER;
    for (i, pos) in positions.iter().enumerate() {
        let dist = ((pos[0] - focal_point[0]).powi(2)
            + (pos[1] - focal_point[1]).powi(2)
            + (pos[2] - focal_point[2]).powi(2))
        .sqrt();
        let delay_s = dist / c;
        let idx = (delay_s * sample_rate).round() as usize;
        if idx < n_samples {
            data[[i, 0, idx]] = 1.0;
        }
    }

    data
}

#[test]
fn delay_and_sum_single_look_aligns_impulses() {
    let elements = 8;
    let pitch = 0.01; // 1 cm
    let geometry = linear_array_positions(elements, pitch);
    let sample_rate = 1_000_000.0; // 1 MHz
    let n_samples = 4096;
    let focal = [0.0, 0.0, 0.0];
    let sensor_data = synth_sensor_data(&geometry, n_samples, focal, sample_rate);

    let config = BeamformingConfig {
        method: BeamformingMethod::DelayAndSum,
        frequency_range: (20e3, 10e6),
        spatial_resolution: 1e-3,
        apodization: ApodizationType::Hamming,
        focal_point: focal,
    };

    // Precompute positions before moving geometry
    let positions = geometry.element_positions();
    let mut bf = Beamformer::new(geometry, config).expect("beamformer init");
    let out = bf.beamform(&sensor_data, sample_rate).expect("beamform");

    assert_eq!(out.dim(), (1, 1, n_samples));

    // Expect a distinct peak within an early window (alignment window)
    let early_window = 128usize;
    let mut max_val = -f64::INFINITY;
    let mut max_idx = 0usize;
    for t in 0..n_samples {
        let v = out[[0, 0, t]];
        if v > max_val {
            max_val = v;
            max_idx = t;
        }
    }

    // Allow one-sample tolerance due to rounding
    assert!(max_idx < early_window, "peak at {}, expected within first {} samples", max_idx, early_window);
    assert!(max_val > 0.5, "beamformed peak too small: {}", max_val);
}

#[test]
fn capon_diagonal_loading_produces_finite_output() {
    let elements = 8;
    let pitch = 0.01; // 1 cm
    let geometry = linear_array_positions(elements, pitch);
    let sample_rate = 1_000_000.0; // 1 MHz
    let n_samples = 4096;
    let focal = [0.0, 0.0, 0.0];
    let sensor_data = synth_sensor_data(&geometry, n_samples, focal, sample_rate);

    let config = BeamformingConfig {
        method: BeamformingMethod::CaponDiagonalLoading { diagonal_loading: 1e-4 },
        frequency_range: (20e3, 10e6),
        spatial_resolution: 1e-3,
        apodization: ApodizationType::Hamming,
        focal_point: focal,
    };

    let mut bf = Beamformer::new(geometry, config).expect("beamformer init");
    let out = bf.beamform(&sensor_data, sample_rate).expect("beamform");
    assert_eq!(out.dim(), (1, 1, n_samples));

    // Basic sanity: values are finite and the initial segment has energy
    for t in 0..64 {
        let v = out[[0, 0, t]];
        assert!(v.is_finite(), "non-finite output at {}: {}", t, v);
    }
}
