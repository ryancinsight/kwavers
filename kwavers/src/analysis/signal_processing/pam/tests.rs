use super::*;
use crate::core::constants::{SAMPLING_FREQUENCY_DEFAULT, SOUND_SPEED_TISSUE};

#[test]
fn pam_policy_to_core_capon_loading_and_midpoint_frequency() {
    let pam = PamBeamformingConfig {
        core: BeamformingCoreConfig::default(),
        method: PamBeamformingMethod::CaponDiagonalLoading {
            diagonal_loading: 0.05,
        },
        frequency_range: (1.0e6, 3.0e6),
        spatial_resolution: 1e-3,
        apodization: ApodizationType::Hamming,
        focal_point: [0.0, 0.0, 0.0],
    };

    let core: BeamformingCoreConfig = pam.into();
    assert!((core.reference_frequency - 2.0e6).abs() < 1.0);
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
        frequency_range: (2.0e6, 2.0e6),
        spatial_resolution: 1e-3,
        apodization: ApodizationType::None,
        focal_point: [0.0, 0.0, 0.0],
    };

    let core: BeamformingCoreConfig = pam.into();
    assert!((core.reference_frequency - 2.0e6).abs() < 1.0);
    assert!((core.diagonal_loading - embedded_core.diagonal_loading).abs() < 1e-12);
}
