//! PAM beamforming configuration (PAM-owned policy layer over shared beamforming SSOT)
//!
//! # Architectural Intent (Deep Vertical, SSOT)
//! - **Algorithms & numerics live in** `crate::sensor::beamforming` (single source of truth).
//! - **PAM owns only policy and PAM-specific knobs** (band selection, focal point, apodization),
//!   and converts deterministically into `BeamformingCoreConfig`.
//!
//! This file exists to prevent PAM from re-implementing beamforming algorithms while still
//! allowing PAM to select *which* shared algorithm to apply.
//!
//! # Invariants
//! - `frequency_range`: must be finite, non-negative, and `f_min <= f_max`.
//! - `focal_point`: all components must be finite.
//! - `spatial_resolution`: must be finite and strictly positive.
//!
//! # Notes
//! - PAM sweeps focal points externally to build maps; this config expresses a **single look**
//!   per invocation to keep downstream math well-defined.
//! - The shared beamforming core config does **not** carry apodization or focal point; those are
//!   consumer-level concerns.

use crate::error::{KwaversError, KwaversResult};
use crate::sensor::beamforming::BeamformingCoreConfig;

/// Beamforming method selection for PAM.
///
/// This selects which shared beamforming algorithm(s) PAM asks the shared
/// beamforming stack to run.
///
/// PAM-specific `TimeExposureAcoustics` is **not** a beamformer; it is a mapping
/// post-processing operator applied to delay-and-sum output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PamBeamformingMethod {
    /// Conventional delay-and-sum.
    DelayAndSum,
    /// MVDR/Capon with diagonal loading.
    ///
    /// This method requires covariance estimation and is sensitive to snapshot policy.
    CaponDiagonalLoading { diagonal_loading: f64 },
    /// MUSIC pseudospectrum (narrowband).
    ///
    /// The implementation lives in the shared beamforming/adaptive stack.
    /// PAM uses this selection when constructing localization-style maps.
    Music { num_sources: usize },
    /// Eigenspace MVDR (E-MVDR / ESMV).
    ///
    /// The implementation lives in the shared beamforming/adaptive stack.
    EigenspaceMinVariance { signal_subspace_dimension: usize },
    /// Time Exposure Acoustics (TEA): integrate squared DAS output over time.
    ///
    /// This is PAM-owned post-processing; the underlying beamforming is still DAS.
    TimeExposureAcoustics,
}

/// Apodization window policy.
///
/// This is PAM-owned (and imaging-owned) policy. The shared beamforming core
/// stays apodization-agnostic and consumes explicit weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApodizationType {
    None,
    Hamming,
    Hanning,
    Blackman,
    Kaiser { beta: f64 },
}

/// PAM-owned configuration used to select and parameterize beamforming for PAM.
///
/// This is a policy wrapper around `BeamformingCoreConfig`.
#[derive(Debug, Clone)]
pub struct PamBeamformingConfig {
    /// Core (shared) configuration: sound speed, sampling frequency, default loading, etc.
    pub core: BeamformingCoreConfig,

    /// Beamforming policy.
    pub method: PamBeamformingMethod,

    /// Frequency range of interest for PAM processing (Hz).
    ///
    /// Used to determine a deterministic reference frequency when needed.
    pub frequency_range: (f64, f64),

    /// Spatial resolution (m). Used by PAM map construction; not required by the shared core.
    pub spatial_resolution: f64,

    /// Apodization of array elements prior to summation / covariance estimation.
    pub apodization: ApodizationType,

    /// Focal point for **single-look** beamforming in meters `[x, y, z]`.
    pub focal_point: [f64; 3],
}

impl PamBeamformingConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> KwaversResult<()> {
        let (f_min, f_max) = self.frequency_range;

        if !(f_min.is_finite() && f_max.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: frequency_range must be finite".to_string(),
            ));
        }
        if f_min < 0.0 || f_max < 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: frequency_range must be non-negative".to_string(),
            ));
        }
        if f_min > f_max {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: require f_min <= f_max".to_string(),
            ));
        }

        if !self.spatial_resolution.is_finite() || self.spatial_resolution <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: spatial_resolution must be finite and > 0".to_string(),
            ));
        }

        if self.focal_point.iter().any(|v| !v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: focal_point must be finite".to_string(),
            ));
        }

        match self.method {
            PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } => {
                if !diagonal_loading.is_finite() || diagonal_loading < 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: diagonal_loading must be finite and >= 0"
                            .to_string(),
                    ));
                }
            }
            PamBeamformingMethod::Music { num_sources } => {
                if num_sources == 0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: MUSIC requires num_sources >= 1".to_string(),
                    ));
                }
            }
            PamBeamformingMethod::EigenspaceMinVariance {
                signal_subspace_dimension,
            } => {
                if signal_subspace_dimension == 0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: ESMV requires signal_subspace_dimension >= 1"
                            .to_string(),
                    ));
                }
            }
            PamBeamformingMethod::DelayAndSum | PamBeamformingMethod::TimeExposureAcoustics => {}
        }

        Ok(())
    }

    /// Deterministic reference frequency derived from `frequency_range`.
    ///
    /// This is suitable for steering or narrowband approximations and is used to
    /// populate `core.reference_frequency` when the caller wishes to enforce it.
    #[must_use]
    pub fn reference_frequency_midpoint(&self) -> f64 {
        let (f_min, f_max) = self.frequency_range;
        0.5 * (f_min + f_max)
    }
}

impl Default for PamBeamformingConfig {
    fn default() -> Self {
        // Keep defaults coherent with the shared core defaults, but choose a
        // PAM-reasonable band and spatial resolution.
        Self {
            core: BeamformingCoreConfig::default(),
            method: PamBeamformingMethod::DelayAndSum,
            frequency_range: (20e3, 10e6),
            spatial_resolution: 1e-3,
            apodization: ApodizationType::Hamming,
            focal_point: [0.0, 0.0, 0.0],
        }
    }
}

/// NOTE: Conversion into `BeamformingCoreConfig` is implemented in
/// `crate::sensor::beamforming::config` to keep SSOT for core configuration conversions.
///
/// This avoids duplicate/conflicting `From` impls and keeps the dependency direction clean:
/// PAM defines policy; beamforming core defines conversions into core configuration.
const _SSOT_CONVERSION_NOTE: () = ();
