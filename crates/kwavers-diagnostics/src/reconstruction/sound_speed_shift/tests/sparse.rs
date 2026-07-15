//! Sparse sampling and L1 proximal localization tests.

use leto::Array2;

use super::{
    attach_time_shifts, horizontal_sample, predict_sound_speed_time_shifts,
    reconstruct_sound_speed_shift, vertical_sample, ShiftPrior, ShiftSampling,
    SoundSpeedShiftConfig,
};

/// Sparse sampling selects every-other row; L1 prior concentrates the
/// reconstruction at the single nonzero cell (2,2) after crossing rays
/// are provided.
#[test]
fn sparse_sampling_and_prior_localize_crossing_shift() {
    let mask = Array2::from_elem((5, 5), true);
    let mut truth = Array2::zeros((5, 5));
    truth[[2, 2]] = 60.0;
    let samples = vec![
        horizontal_sample(0.0),
        horizontal_sample(-0.002),
        vertical_sample(0.0),
        vertical_sample(-0.002),
    ];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        iterations: 160,
        tikhonov_weight: 0.0,
        smoothness_weight: 0.0,
        sparsity_weight: 1.0e-5,
        sampling: ShiftSampling::Sparse {
            stride: 2,
            offset: 0,
        },
        prior: ShiftPrior::Sparse,
        ..Default::default()
    };
    let predicted = predict_sound_speed_time_shifts(&truth, &samples, &mask, config).unwrap();
    let measured = attach_time_shifts(&samples, &predicted);

    let image = reconstruct_sound_speed_shift(&measured, &mask, config).unwrap();
    let center = image.sound_speed_shift_m_s[[2, 2]];
    let neighbor = image.sound_speed_shift_m_s[[2, 1]].max(image.sound_speed_shift_m_s[[1, 2]]);

    assert_eq!(image.rows_available, 4);
    assert_eq!(image.rows_used, 2);
    assert!(center > 0.0, "center perturbation was not recovered");
    assert!(
        center > neighbor,
        "sparse crossing row should give center dominance, center={center}, neighbor={neighbor}"
    );
}

/// Zero-stride sparse sampling config is rejected with a descriptive error.
#[test]
fn invalid_sparse_sampling_is_rejected() {
    let mask = Array2::from_elem((1, 1), true);
    let samples = vec![horizontal_sample(0.0)];
    let config = SoundSpeedShiftConfig {
        spacing_m: 0.001,
        sampling: ShiftSampling::Sparse {
            stride: 0,
            offset: 0,
        },
        ..Default::default()
    };

    let err = reconstruct_sound_speed_shift(&samples, &mask, config).unwrap_err();
    assert!(err.to_string().contains("Sparse sampling requires stride"));
}
