//! Integration tests for PAM beamforming that enforce SSOT usage.
//!
//! Invariants:
//! - PAM must not re-implement beamforming algorithms; it must compose `sensor::beamforming`.
//! - Tests must exercise only what we touched: PAM + shared `BeamformingProcessor` integration.
//! - Tests must run in `--release` and be deterministic.

use kwavers::domain::sensor::beamforming::BeamformingCoreConfig;
use kwavers::domain::sensor::passive_acoustic_mapping::geometry::ArrayGeometry;
use kwavers::domain::sensor::passive_acoustic_mapping::{
    ApodizationType, PAMConfig, PamBeamformingConfig, PamBeamformingMethod, PassiveAcousticMapper,
};
use kwavers::physics::constants::SOUND_SPEED_WATER;
use ndarray::{Array3, Axis};

fn linear_array_positions(elements: usize, pitch_m: f64) -> ArrayGeometry {
    ArrayGeometry::Linear {
        elements,
        pitch: pitch_m,
        center: [0.0, 0.0, 0.0],
        orientation: [1.0, 0.0, 0.0],
    }
}

fn synth_sensor_data_impulses(
    geometry: &ArrayGeometry,
    n_samples: usize,
    focal_point: [f64; 3],
    sample_rate: f64,
    sound_speed: f64,
) -> Array3<f64> {
    // Shape: (n_elements, 1, n_samples)
    let n_elements = geometry.num_elements();
    let mut data = Array3::<f64>::zeros((n_elements, 1, n_samples));

    let positions = geometry.element_positions();
    for (i, pos) in positions.iter().enumerate() {
        let dx = pos[0] - focal_point[0];
        let dy = pos[1] - focal_point[1];
        let dz = pos[2] - focal_point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        let delay_s = dist / sound_speed;
        let idx = (delay_s * sample_rate).round() as isize;

        if idx >= 0 && (idx as usize) < n_samples {
            data[[i, 0, idx as usize]] = 1.0;
        }
    }

    data
}

fn pam_config_for(
    method: PamBeamformingMethod,
    focal_point: [f64; 3],
    sample_rate: f64,
    sound_speed: f64,
) -> PAMConfig {
    let core = BeamformingCoreConfig {
        sampling_frequency: sample_rate,
        sound_speed,
        ..Default::default()
    };

    let beamforming = PamBeamformingConfig {
        core,
        method,
        frequency_range: (20e3, 10e6),
        spatial_resolution: 1e-3,
        apodization: ApodizationType::Hamming,
        focal_point,
    };

    PAMConfig {
        beamforming,
        // Keep bands minimal so tests focus on integration not DSP decisions.
        frequency_bands: vec![(20e3, 100e3)],
        integration_time: 0.1,
        threshold: 0.0, // accept any non-zero power
        enable_harmonic_analysis: false,
        enable_broadband_analysis: false,
    }
}

#[test]
fn pam_delay_and_sum_pipeline_produces_nonzero_map() {
    let elements = 8;
    let pitch = 0.01;
    let geometry = linear_array_positions(elements, pitch);

    let sample_rate = 1_000_000.0; // 1 MHz
    let n_samples = 2048;
    let focal = [0.0, 0.0, 0.0];
    let sound_speed = SOUND_SPEED_WATER;

    let sensor_data =
        synth_sensor_data_impulses(&geometry, n_samples, focal, sample_rate, sound_speed);

    let config = pam_config_for(
        PamBeamformingMethod::DelayAndSum,
        focal,
        sample_rate,
        sound_speed,
    );
    let mut pam = PassiveAcousticMapper::new(config, geometry).expect("PAM init");

    let map = pam
        .process(&sensor_data, sample_rate)
        .expect("PAM process (DAS)");

    // Output: (nx, ny, nbands). For the current PAM pipeline, beamformed data is (1,1,nt),
    // so map is expected to be (1,1,nbands).
    assert_eq!(map.shape()[0], 1);
    assert_eq!(map.shape()[1], 1);
    assert_eq!(map.shape()[2], 1);

    let v = map[[0, 0, 0]];
    assert!(v.is_finite());
    assert!(v > 0.0, "expected non-zero band power, got {v}");
}

#[test]
fn pam_capon_pipeline_produces_finite_map() {
    let elements = 8;
    let pitch = 0.01;
    let geometry = linear_array_positions(elements, pitch);

    let sample_rate = 1_000_000.0; // 1 MHz
    let n_samples = 2048;
    let focal = [0.0, 0.0, 0.0];
    let sound_speed = SOUND_SPEED_WATER;

    let sensor_data =
        synth_sensor_data_impulses(&geometry, n_samples, focal, sample_rate, sound_speed);

    let config = pam_config_for(
        PamBeamformingMethod::CaponDiagonalLoading {
            diagonal_loading: 1e-4,
        },
        focal,
        sample_rate,
        sound_speed,
    );
    let mut pam = PassiveAcousticMapper::new(config, geometry).expect("PAM init");

    let map = pam
        .process(&sensor_data, sample_rate)
        .expect("PAM process (Capon)");

    assert_eq!(map.shape()[0], 1);
    assert_eq!(map.shape()[1], 1);
    assert_eq!(map.shape()[2], 1);

    let v = map[[0, 0, 0]];
    assert!(v.is_finite(), "non-finite map value: {v}");
    assert!(v >= 0.0, "band power must be non-negative: {v}");
}

#[test]
fn pam_time_exposure_acoustics_outputs_single_time_plane() {
    let elements = 8;
    let pitch = 0.01;
    let geometry = linear_array_positions(elements, pitch);

    let sample_rate = 1_000_000.0;
    let n_samples = 2048;
    let focal = [0.0, 0.0, 0.0];
    let sound_speed = SOUND_SPEED_WATER;

    // Use impulses; TEA should integrate (DAS^2) over time => positive scalar.
    let sensor_data =
        synth_sensor_data_impulses(&geometry, n_samples, focal, sample_rate, sound_speed);

    let config = pam_config_for(
        PamBeamformingMethod::TimeExposureAcoustics,
        focal,
        sample_rate,
        sound_speed,
    );
    let mut pam = PassiveAcousticMapper::new(config, geometry).expect("PAM init");

    let map = pam
        .process(&sensor_data, sample_rate)
        .expect("PAM process (TEA)");

    assert_eq!(map.shape(), &[1, 1, 1]);

    let v = map[[0, 0, 0]];
    assert!(v.is_finite());
    assert!(
        v > 0.0,
        "TEA should be strictly positive for impulse-aligned data"
    );
}

/// Sanity check: the synthetic impulse data has exactly one impulse per element.
#[test]
fn synth_impulses_has_one_impulse_per_element() {
    let elements = 8;
    let pitch = 0.01;
    let geometry = linear_array_positions(elements, pitch);

    let sample_rate = 1_000_000.0;
    let n_samples = 2048;
    let focal = [0.0, 0.0, 0.0];

    let data =
        synth_sensor_data_impulses(&geometry, n_samples, focal, sample_rate, SOUND_SPEED_WATER);

    // Count impulses per element channel.
    for elem in 0..elements {
        let channel = data.index_axis(Axis(0), elem);
        let mut count = 0usize;
        for v in channel.iter() {
            if *v == 1.0 {
                count += 1;
            } else {
                // enforce exact 0/1 outputs from the generator
                assert!(*v == 0.0, "unexpected non-binary sample: {v}");
            }
        }
        assert_eq!(
            count, 1,
            "expected 1 impulse for element {elem}, got {count}"
        );
    }
}
