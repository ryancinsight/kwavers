use super::*;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::{SAMPLING_FREQUENCY_DEFAULT, SOUND_SPEED_TISSUE};

#[test]
fn pam_policy_to_core_capon_loading_and_midpoint_frequency() {
    let pam = PamBeamformingConfig {
        core: BeamformingCoreConfig::default(),
        method: PamBeamformingMethod::CaponDiagonalLoading {
            diagonal_loading: 0.05,
        },
        frequency_range: (MHZ_TO_HZ, 3.0 * MHZ_TO_HZ),
        spatial_resolution: 1e-3,
        apodization: ApodizationType::Hamming,
        focal_point: [0.0, 0.0, 0.0],
    };

    let core: BeamformingCoreConfig = pam.into();
    assert!((core.reference_frequency - 2.0 * MHZ_TO_HZ).abs() < 1.0);
    assert!((core.diagonal_loading - 0.05).abs() < 1e-12);

    assert_eq!(core.sound_speed, SOUND_SPEED_TISSUE);
    assert_eq!(core.sampling_frequency, SAMPLING_FREQUENCY_DEFAULT);
    assert_eq!(core.num_snapshots, 100);
    assert_eq!(core.spatial_smoothing, None);
}

#[test]
fn pam_policy_to_core_non_capon_preserves_core_loading_and_sets_reference_frequency() {
    let embedded_core = BeamformingCoreConfig {
        diagonal_loading: 0.123,
        ..Default::default()
    };

    let pam = PamBeamformingConfig {
        core: embedded_core.clone(),
        method: PamBeamformingMethod::DelayAndSum,
        frequency_range: (2.0 * MHZ_TO_HZ, 2.0 * MHZ_TO_HZ),
        spatial_resolution: 1e-3,
        apodization: ApodizationType::Uniform,
        focal_point: [0.0, 0.0, 0.0],
    };

    let core: BeamformingCoreConfig = pam.into();
    assert!((core.reference_frequency - 2.0 * MHZ_TO_HZ).abs() < 1.0);
    assert!((core.diagonal_loading - embedded_core.diagonal_loading).abs() < 1e-12);
}

#[test]
fn eigenspace_covariance_eigenvalues_pin_signal_noise_split() {
    let eigenvalues = eigenspace_covariance_eigenvalues(8, 3, 10.0, 1.0).unwrap();

    assert_eq!(eigenvalues.len(), 8);
    assert_eq!(&eigenvalues[..3], &[11.0, 11.0, 11.0]);
    assert_eq!(&eigenvalues[3..], &[1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn eigenspace_covariance_eigenvalues_reject_invalid_inputs() {
    assert_eq!(
        eigenspace_covariance_eigenvalues(0, 1, 10.0, 1.0),
        Err(EigenspaceSpectrumError::EmptyAperture)
    );
    assert_eq!(
        eigenspace_covariance_eigenvalues(8, 0, 10.0, 1.0),
        Err(EigenspaceSpectrumError::InvalidSourceRank {
            n_elements: 8,
            n_sources: 0,
        })
    );
    assert_eq!(
        eigenspace_covariance_eigenvalues(8, 8, 10.0, 1.0),
        Err(EigenspaceSpectrumError::InvalidSourceRank {
            n_elements: 8,
            n_sources: 8,
        })
    );
    match eigenspace_covariance_eigenvalues(8, 3, f64::NAN, 1.0) {
        Err(EigenspaceSpectrumError::InvalidSignalPower { signal_power }) => {
            assert!(signal_power.is_nan());
        }
        other => panic!("expected invalid signal power error, got {other:?}"),
    }
    assert_eq!(
        eigenspace_covariance_eigenvalues(8, 3, 10.0, 0.0),
        Err(EigenspaceSpectrumError::InvalidNoisePower { noise_power: 0.0 })
    );
}
