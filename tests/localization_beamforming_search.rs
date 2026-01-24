//! Release tests for SSOT-compliant beamforming-based localization.
//!
//! This test validates the dedicated API:
//! - `LocalizationProcessor::localize_beamforming`
//! - `localization::beamforming_search::{BeamformSearch, LocalizationBeamformSearchConfig, ...}`
//!
//! Key invariants enforced:
//! - Localization does not re-implement beamforming math.
//! - The API consumes raw time-series sensor data shaped `(n_sensors, 1, n_samples)`.
//! - Search config sampling frequency must match the input sampling frequency.
//!
//! # Field jargon / capability coverage
//! - **A (transient/broadband)**: SRP-DAS (Steered Response Power using time-domain DAS).
//!   This requires an explicit **delay datum / reference**. The most common deterministic choice
//!   is the reference sensor at index 0 (`SensorIndex(0)`), which is the SSOT recommended default.
//! - **B (narrowband/adaptive)** is covered by separate tests (Capon/MVDR spatial spectrum), not here.

use kwavers::analysis::signal_processing::beamforming::time_domain::DelayReference;
use kwavers::domain::sensor::beamforming::BeamformingCoreConfig;
use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor};
use kwavers::domain::sensor::localization::{
    BeamformingLocalizationInput, LocalizationBeamformSearchConfig, LocalizationBeamformingMethod,
    LocalizationConfig, LocalizationMethod, LocalizationProcessor, Position, SearchGrid,
    SensorArray,
};

use ndarray::Array3;

fn make_array(sound_speed: f64) -> SensorArray {
    // Small 2D aperture in the xy-plane, z=0.
    // Units: meters.
    let sensors = vec![
        Sensor::new(0, Position::new(-0.015, 0.0, 0.0)),
        Sensor::new(1, Position::new(-0.005, 0.0, 0.0)),
        Sensor::new(2, Position::new(0.005, 0.0, 0.0)),
        Sensor::new(3, Position::new(0.015, 0.0, 0.0)),
    ];

    SensorArray::new(sensors, sound_speed, ArrayGeometry::Arbitrary)
}

/// Synthesize transient impulse data consistent with SRP-DAS and an explicit delay reference.
///
/// Model:
/// - Let `τ_i(p)` be absolute TOF from source to sensor i.
/// - Choose a delay datum `τ_ref = τ_ref(p)` using `delay_reference`.
/// - Place impulses at indices proportional to **relative** delays: `Δτ_i = τ_i - τ_ref`.
///
/// This avoids silent conventions (e.g., latest-arrival normalization) and ensures the SRP-DAS
/// score is point-dependent under the intended model.
fn synth_impulse_sensor_data_with_delay_reference(
    array: &SensorArray,
    source: Position,
    sample_rate: f64,
    n_samples: usize,
    delay_reference: DelayReference,
    emission_index: usize,
) -> Array3<f64> {
    // Shape: (n_sensors, 1, n_samples)
    let n_sensors = array.num_sensors();
    let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));

    let c = array.sound_speed();

    // Compute absolute TOFs.
    let mut delays_s: Vec<f64> = Vec::with_capacity(n_sensors);
    for i in 0..n_sensors {
        let sensor_pos = array.get_sensor_position(i);
        let d = source.distance_to(sensor_pos);
        delays_s.push(d / c);
    }

    // Resolve τ_ref according to the explicit policy (match SSOT behavior).
    let tau_ref = match delay_reference {
        DelayReference::SensorIndex(idx) => delays_s[idx],
        DelayReference::EarliestArrival => delays_s.iter().copied().fold(f64::INFINITY, f64::min),
        DelayReference::LatestArrival => delays_s.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    };

    // Emit impulses at t = emission_index + round((τ_i - τ_ref) * fs).
    // This centers the event away from t=0 so negative/positive relative delays stay in-bounds.
    for i in 0..n_sensors {
        let rel_s = delays_s[i] - tau_ref;
        let shift = (rel_s * sample_rate).round() as isize;
        let idx = emission_index as isize + shift;

        if idx >= 0 && (idx as usize) < n_samples {
            data[[i, 0, idx as usize]] = 1.0;
        }
    }

    data
}

#[test]
fn beamforming_localization_finds_source_near_true_position() {
    // Keep this test deterministic and fast in release.
    let sound_speed = 1500.0;
    let sample_rate = 1_000_000.0;
    let n_samples = 2048;

    let array = make_array(sound_speed);

    // True source within the search cube.
    let true_source = Position::new(0.0, 0.01, 0.0);

    // Field-jargon correct default for SRP-DAS: fixed delay datum at reference sensor 0.
    let delay_reference = DelayReference::recommended_default();

    // Place the emission away from the start so relative delays stay in-bounds.
    let sensor_data = synth_impulse_sensor_data_with_delay_reference(
        &array,
        true_source,
        sample_rate,
        n_samples,
        delay_reference,
        256,
    );

    let loc_cfg = LocalizationConfig {
        sound_speed,
        max_iterations: 100,
        tolerance: 1e-6,
        use_gpu: false,
        method: LocalizationMethod::Beamforming,
        frequency: 1e6,
        search_radius: Some(0.05),
    };

    let processor = LocalizationProcessor::new(loc_cfg, array);

    let mut core = BeamformingCoreConfig::default();
    core.sound_speed = sound_speed;
    core.sampling_frequency = sample_rate;
    core.reference_frequency = 1e6;

    let search_cfg = LocalizationBeamformSearchConfig {
        core,
        method: LocalizationBeamformingMethod::SrpDasTimeDomain { delay_reference },
        grid: SearchGrid::CenteredCube {
            search_radius_m: 0.05,
            grid_resolution_m: 0.01,
            min_points_per_axis: 7,
        },
        normalize_by_sensor_count: true,
    };

    let input = BeamformingLocalizationInput {
        sensor_data,
        sampling_frequency: sample_rate,
    };

    let result = processor
        .localize_beamforming(&input, search_cfg)
        .expect("beamforming localization (SRP-DAS) should succeed");

    // The grid resolution is 1 cm, accept within ~1.5 grid steps.
    let err = result.position.distance_to(&true_source);
    assert!(
        err <= 0.02 + 1e-12,
        "expected localization error <= 2cm; got {err} (estimated={:?}, true={:?})",
        result.position,
        true_source
    );
}

#[test]
fn beamforming_localization_rejects_sampling_frequency_mismatch() {
    let sound_speed = 1500.0;
    let sample_rate = 1_000_000.0;
    let n_samples = 512;

    let array = make_array(sound_speed);
    let true_source = Position::new(0.0, 0.01, 0.0);

    let delay_reference = DelayReference::recommended_default();
    let sensor_data = synth_impulse_sensor_data_with_delay_reference(
        &array,
        true_source,
        sample_rate,
        n_samples,
        delay_reference,
        64,
    );

    let loc_cfg = LocalizationConfig {
        sound_speed,
        max_iterations: 100,
        tolerance: 1e-6,
        use_gpu: false,
        method: LocalizationMethod::Beamforming,
        frequency: 1e6,
        search_radius: Some(0.05),
    };

    let processor = LocalizationProcessor::new(loc_cfg, array);

    let mut core = BeamformingCoreConfig::default();
    core.sound_speed = sound_speed;
    core.sampling_frequency = sample_rate + 1.0; // deliberate mismatch
    core.reference_frequency = 1e6;

    let search_cfg = LocalizationBeamformSearchConfig {
        core,
        method: LocalizationBeamformingMethod::SrpDasTimeDomain { delay_reference },
        grid: SearchGrid::CenteredCube {
            search_radius_m: 0.05,
            grid_resolution_m: 0.01,
            min_points_per_axis: 5,
        },
        normalize_by_sensor_count: true,
    };

    let input = BeamformingLocalizationInput {
        sensor_data,
        sampling_frequency: sample_rate,
    };

    let err = processor
        .localize_beamforming(&input, search_cfg)
        .expect_err("expected sampling frequency mismatch to be rejected");

    let msg = err.to_string();
    assert!(
        msg.contains("sampling_frequency"),
        "expected error mentioning sampling_frequency; got: {msg}"
    );
}
