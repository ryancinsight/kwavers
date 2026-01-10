//! Beamforming configuration (core, unified)

use crate::domain::core::constants::{SAMPLING_FREQUENCY_DEFAULT, SOUND_SPEED_TISSUE};

/// Core configuration for beamforming operations across array-processing consumers.
///
/// This struct is the single source of truth for physical and numerical
/// parameters used by array beamforming. Downstream modules (e.g., PAM and
/// localization) should wrap or convert into this core type.
///
/// The legacy name `BeamformingConfig` remains available as a type alias to
/// preserve API stability while consolidating the architecture.
#[doc(alias = "BeamformingConfig")]
#[derive(Debug, Clone)]
pub struct BeamformingCoreConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Reference frequency for array design (Hz)
    pub reference_frequency: f64,
    /// Diagonal loading factor for regularization
    pub diagonal_loading: f64,
    /// Number of snapshots for covariance estimation
    pub num_snapshots: usize,
    /// Spatial smoothing factor
    pub spatial_smoothing: Option<usize>,
}

impl Default for BeamformingCoreConfig {
    fn default() -> Self {
        const REFERENCE_FREQUENCY: f64 = 5e6; // 5 MHz
        const DIAGONAL_LOADING_FACTOR: f64 = 0.01; // 1% diagonal loading
        const DEFAULT_SNAPSHOTS: usize = 100;

        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            sampling_frequency: SAMPLING_FREQUENCY_DEFAULT,
            reference_frequency: REFERENCE_FREQUENCY,
            diagonal_loading: DIAGONAL_LOADING_FACTOR,
            num_snapshots: DEFAULT_SNAPSHOTS,
            spatial_smoothing: None,
        }
    }
}

/// Backward-compatible alias for the unified core configuration.
pub type BeamformingConfig = BeamformingCoreConfig;

/// Deterministic conversion from PAM's **policy** config to the shared core
/// beamforming configuration.
///
/// PAM owns policy knobs (method selection, focal point, apodization, frequency
/// range) while `BeamformingCoreConfig` owns the beamforming numerical/physical
/// primitives (sound speed, sampling frequency, diagonal loading defaults, etc.).
///
/// Mapping rationale:
/// - `reference_frequency`: set to the midpoint of PAM `frequency_range`.
/// - `diagonal_loading`: if PAM selects Capon with loading, adopt the provided value;
///   otherwise preserve the `core.diagonal_loading`.
/// - Remaining fields are preserved from PAM's embedded `core` to avoid overriding
///   caller-specified physical parameters.
impl From<crate::domain::sensor::passive_acoustic_mapping::beamforming_config::PamBeamformingConfig>
    for BeamformingCoreConfig
{
    fn from(
        pam: crate::domain::sensor::passive_acoustic_mapping::beamforming_config::PamBeamformingConfig,
    ) -> Self {
        use crate::domain::sensor::passive_acoustic_mapping::beamforming_config::PamBeamformingMethod;

        let (f_min, f_max) = pam.frequency_range;
        let reference_frequency = 0.5 * (f_min + f_max);

        let mut core = pam.core;
        core.reference_frequency = reference_frequency;

        if let PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } = pam.method {
            core.diagonal_loading = diagonal_loading;
        }

        core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::sensor::passive_acoustic_mapping::beamforming_config::{
        ApodizationType, PamBeamformingConfig, PamBeamformingMethod,
    };

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

        // These should remain aligned to the SSOT defaults unless the caller overrides `pam.core`.
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
}
