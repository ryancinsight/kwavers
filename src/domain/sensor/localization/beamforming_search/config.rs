//! Localization beamforming-search configuration (policy layer over shared beamforming SSOT).
//!
//! # Architectural intent (Deep Vertical, SSOT)
//! - Beamforming algorithms and numerical primitives are owned by `crate::sensor::beamforming`.
//! - Localization owns only *policy* for search: grid definition, resolution, and which shared
//!   algorithm to apply for scoring.
//!
//! This module exists to eliminate redundant beamforming implementations inside localization and
//! ensure localization composes the shared beamforming stack.
//!
//! # Field jargon / capabilities
//! - **A (broadband / transient)**: **SRP-DAS** (Steered Response Power with time-domain DAS).
//!   Candidate points are scored via a delay datum / delay reference (recommended default: reference
//!   sensor 0) and an energy functional such as `∑_t y_p(t)^2`.
//! - **B (narrowband / adaptive)**: **Capon/MVDR spatial spectrum** (a.k.a. MVDR spectrum):
//!   `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))`, where `a(p)` is a look-dependent steering vector and
//!   `R` is the sample covariance with optional diagonal loading.
//!
//! # Invariants
//! - `search_radius_m > 0` (when used).
//! - `grid_resolution_m > 0`.
//! - `points_m` (if provided) is non-empty and finite.
//! - All distances, frequencies, and loading factors must be finite (with appropriate sign).
//!
//! # Notes
//! - This configuration intentionally does not define beamforming math; steering/delays and adaptive
//!   algorithms are computed by `crate::sensor::beamforming` (SSOT).
//! - This config is compatible with implementations that score candidate points without copying or
//!   re-implementing array-processing numerics in localization.
//!
//! # Covariance domain selection (MVDR/Capon correctness)
//! MVDR/Capon and subspace methods are mathematically native to **complex baseband** snapshots with a
//! **Hermitian** covariance `R = (1/K) ∑ x_k x_kᴴ`. However, a pragmatic baseline is to estimate a
//! **real** covariance directly from real time samples.
//!
//! This module makes the covariance/snapshot domain an explicit policy choice for localization
//! scoring to prevent silent fallback behavior.

use crate::core::error::{KwaversError, KwaversResult};
// Import from analysis layer (now canonical location)
use crate::analysis::signal_processing::beamforming::time_domain::DelayReference;
use crate::domain::sensor::beamforming::{BeamformingCoreConfig, SteeringVectorMethod};

/// Covariance / snapshot domain policy for narrowband MVDR/Capon scoring.
///
/// This is a pure policy enum: it does not implement any math.
/// The chosen variant must be honored by orchestrators (e.g., `beamforming_search::BeamformSearch`)
/// without silent fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvdrCovarianceDomain {
    /// Pragmatic baseline: estimate a real covariance from real-valued time samples.
    ///
    /// This is useful for quick diagnostics but is not the mathematically canonical narrowband path.
    RealTimeSamples,
    /// Mathematically canonical narrowband path: complex baseband snapshots with Hermitian covariance.
    ///
    /// Snapshots are extracted by analytic signal (Hilbert transform) + downconversion at `frequency_hz`,
    /// then `R = (1/K) ∑ x_k x_kᴴ` is estimated.
    ///
    /// - `snapshot_step_samples` controls snapshot stride (>= 1).
    ComplexBaseband {
        /// Snapshot stride in samples (>= 1).
        snapshot_step_samples: usize,
    },
}

impl Default for MvdrCovarianceDomain {
    fn default() -> Self {
        // Prefer correctness by default: complex baseband with stride 1.
        Self::ComplexBaseband {
            snapshot_step_samples: 1,
        }
    }
}

/// Beamforming scorer selection for localization grid search.
///
/// This is a *policy* enum: it selects which shared beamforming computation to use for scoring
/// candidate points. Implementations are in `crate::sensor::beamforming`.
///
/// Naming is intentionally field-jargon aligned:
/// - **SRP-DAS**: steered response power using time-domain DAS (transient/broadband).
/// - **Capon/MVDR spectrum**: narrowband adaptive spatial spectrum (point-steered).
#[derive(Debug, Clone, PartialEq)]
pub enum LocalizationBeamformingMethod {
    /// **SRP-DAS** (Steered Response Power with time-domain Delay-and-Sum).
    ///
    /// Uses explicit time-of-flight (TOF) delays and a **delay datum** (`delay_reference`) to define
    /// the alignment convention, then scores energy of the steered output.
    SrpDasTimeDomain {
        /// Delay datum / delay reference policy (recommended default: reference sensor 0).
        delay_reference: DelayReference,
    },

    /// **Capon/MVDR spatial spectrum** (narrowband, adaptive).
    ///
    /// Scores candidate points via `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))` with diagonal loading.
    ///
    /// # Critical policy: covariance/snapshot domain
    /// The covariance domain is explicit to avoid silent fallback behavior:
    /// - `RealTimeSamples`: estimates a real covariance from real-valued time samples.
    /// - `ComplexBaseband`: extracts complex baseband snapshots (analytic signal + downconversion)
    ///   and estimates a Hermitian covariance from `x xᴴ`.
    CaponMvdrSpectrum {
        /// Narrowband frequency (Hz) at which the steering vector is evaluated.
        frequency_hz: f64,
        /// Diagonal loading (δ ≥ 0) added to the covariance matrix for robustness.
        diagonal_loading: f64,
        /// Steering model used to build `a(p)` (near-field: prefer spherical-wave / focused).
        steering: SteeringVectorMethod,
        /// Covariance / snapshot domain policy for MVDR/Capon scoring.
        covariance_domain: MvdrCovarianceDomain,
    },
}

/// Search space definition for beamforming-based localization.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchGrid {
    /// Generate a cubic grid centered on the array centroid with a given radius and resolution.
    ///
    /// Grid spans:
    /// - x ∈ [cx - r, cx + r]
    /// - y ∈ [cy - r, cy + r]
    /// - z ∈ [cz - r, cz + r]
    ///
    /// with step size `grid_resolution_m`.
    CenteredCube {
        /// Radius (m) around centroid.
        search_radius_m: f64,
        /// Step size (m).
        grid_resolution_m: f64,
        /// Minimum number of points per axis (guarantees non-degenerate grid).
        min_points_per_axis: usize,
    },

    /// Provide an explicit list of candidate points (meters).
    ///
    /// This is useful for adaptive refinement or application-specific search sets.
    ExplicitPoints {
        /// Candidate points `[x, y, z]` in meters.
        points_m: Vec<[f64; 3]>,
    },
}

/// Configuration for beamforming grid-search localization.
///
/// This is the localization-owned wrapper around the shared `BeamformingCoreConfig`.
#[derive(Debug, Clone)]
pub struct LocalizationBeamformSearchConfig {
    /// Shared beamforming core configuration (SSOT for physics + numerics).
    pub core: BeamformingCoreConfig,

    /// Scorer used to evaluate each candidate point.
    pub method: LocalizationBeamformingMethod,

    /// Search grid definition.
    pub grid: SearchGrid,

    /// If true, normalize scores by number of sensors to reduce dependence on array size.
    pub normalize_by_sensor_count: bool,
}

impl LocalizationBeamformSearchConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> KwaversResult<()> {
        // Validate shared core invariants relevant to search usage.
        if !self.core.sound_speed.is_finite() || self.core.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LocalizationBeamformSearchConfig: core.sound_speed must be finite and > 0"
                    .to_string(),
            ));
        }
        if !self.core.sampling_frequency.is_finite() || self.core.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LocalizationBeamformSearchConfig: core.sampling_frequency must be finite and > 0"
                    .to_string(),
            ));
        }
        if !self.core.reference_frequency.is_finite() || self.core.reference_frequency < 0.0 {
            return Err(KwaversError::InvalidInput(
                "LocalizationBeamformSearchConfig: core.reference_frequency must be finite and >= 0"
                    .to_string(),
            ));
        }

        match &self.method {
            LocalizationBeamformingMethod::SrpDasTimeDomain { delay_reference: _ } => {
                // Delay reference validation is performed when resolving relative delays against
                // the actual sensor count (runtime), but the enum itself is well-formed.
            }
            LocalizationBeamformingMethod::CaponMvdrSpectrum {
                frequency_hz,
                diagonal_loading,
                steering: _,
                covariance_domain,
            } => {
                if !frequency_hz.is_finite() || *frequency_hz <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "LocalizationBeamformSearchConfig: CaponMvdrSpectrum.frequency_hz must be finite and > 0"
                            .to_string(),
                    ));
                }
                if !diagonal_loading.is_finite() || *diagonal_loading < 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "LocalizationBeamformSearchConfig: CaponMvdrSpectrum.diagonal_loading must be finite and >= 0"
                            .to_string(),
                    ));
                }

                // Ensure the covariance-domain policy is internally consistent.
                match covariance_domain {
                    MvdrCovarianceDomain::RealTimeSamples => {}
                    MvdrCovarianceDomain::ComplexBaseband {
                        snapshot_step_samples,
                    } => {
                        if *snapshot_step_samples == 0 {
                            return Err(KwaversError::InvalidInput(
                                "LocalizationBeamformSearchConfig: CaponMvdrSpectrum.covariance_domain.ComplexBaseband.snapshot_step_samples must be >= 1"
                                    .to_string(),
                            ));
                        }
                    }
                }
            }
        }

        match &self.grid {
            SearchGrid::CenteredCube {
                search_radius_m,
                grid_resolution_m,
                min_points_per_axis,
            } => {
                if !search_radius_m.is_finite() || *search_radius_m <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::CenteredCube: search_radius_m must be finite and > 0"
                            .to_string(),
                    ));
                }
                if !grid_resolution_m.is_finite() || *grid_resolution_m <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::CenteredCube: grid_resolution_m must be finite and > 0"
                            .to_string(),
                    ));
                }
                if *min_points_per_axis < 2 {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::CenteredCube: min_points_per_axis must be >= 2".to_string(),
                    ));
                }
            }
            SearchGrid::ExplicitPoints { points_m } => {
                if points_m.is_empty() {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::ExplicitPoints: points_m must be non-empty".to_string(),
                    ));
                }
                if points_m.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::ExplicitPoints: all coordinates must be finite".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// A conservative default search config intended for small arrays and quick diagnostics.
    ///
    /// This default is intentionally modest: it favors stability over resolution.
    #[must_use]
    pub fn conservative_default() -> Self {
        Self {
            core: BeamformingCoreConfig::default(),
            method: LocalizationBeamformingMethod::SrpDasTimeDomain {
                delay_reference: DelayReference::recommended_default(),
            },
            grid: SearchGrid::CenteredCube {
                search_radius_m: 1.0,
                grid_resolution_m: 0.05,
                min_points_per_axis: 10,
            },
            normalize_by_sensor_count: true,
        }
    }
}

impl Default for LocalizationBeamformSearchConfig {
    fn default() -> Self {
        Self::conservative_default()
    }
}
