#![deny(missing_docs)]
//! Beamforming-based localization (grid search) built on the shared beamforming SSOT.
//!
//! # Architectural Intent (Deep Vertical, SSOT)
//! This module exists to eliminate redundant beamforming implementations inside localization.
//!
//! - **Beamforming algorithms & numerics** are owned by `crate::analysis::signal_processing::beamforming`.
//! - **Localization** owns only *policy* (search grid definition, scorer selection) and orchestration.
//!
//! Concretely, this module provides:
//! - `LocalizationBeamformSearchConfig`: a localization-owned wrapper around `BeamformingCoreConfig`
//! - `BeamformSearch`: a grid-search evaluator that uses shared SSOT beamforming primitives
//!
//! # Field jargon / supported capabilities
//! - **A (broadband / transient)**: **SRP-DAS** (Steered Response Power with time-domain Delay-and-Sum).
//!   Candidate points are scored by steering (TOF alignment using an explicit delay datum / reference)
//!   and evaluating time-domain energy.
//! - **B (narrowband / adaptive)**: point-steered **MVDR/Capon spatial spectrum**.
//!
//! # Data Model Assumption
//! Sensor data is shaped `(n_elements, 1, n_samples)`.
//!
//! # Invariants
//! - No beamforming algorithm math should live in this module.
//! - All candidate point scoring is performed via shared `crate::analysis::signal_processing::beamforming` SSOT APIs.
//! - MVDR covariance-domain selection is an explicit policy choice; no silent fallback is permitted.
//!
//! # MVDR covariance-domain policy (explicit)
//! For MVDR/Capon scoring, the covariance/snapshot domain is an explicit policy choice via
//! `MvdrCovarianceDomain` (e.g., complex baseband snapshots vs. real time-sample covariance).

use crate::analysis::signal_processing::beamforming::domain_processor::BeamformingProcessor;
use crate::analysis::signal_processing::beamforming::time_domain::DelayReference;
use crate::analysis::signal_processing::beamforming::utils::steering::SteeringVectorMethod;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::beamforming::BeamformingCoreConfig;
use crate::domain::sensor::localization::{LocalizationMethod, LocalizationResult, Position, SensorArray};
use ndarray::Array3;

/// Covariance / snapshot domain policy for narrowband MVDR/Capon scoring.
///
/// This is a pure policy enum: it does not implement any math.
/// The chosen variant must be honored by orchestrators (e.g., `BeamformSearch`)
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
        Self::ComplexBaseband {
            snapshot_step_samples: 1,
        }
    }
}

/// Beamforming scorer selection for localization grid search.
///
/// This is a *policy* enum: it selects which shared beamforming computation to use for scoring
/// candidate points. Implementations are in `crate::analysis::signal_processing::beamforming`.
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
            LocalizationBeamformingMethod::SrpDasTimeDomain { delay_reference: _ } => {}
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

/// Dedicated input type for beamforming-based localization.
///
/// This is required because SSOT-compliant beamforming requires raw time-series data:
/// shape `(n_sensors, 1, n_samples)`. Scalar per-sensor measurements are insufficient
/// for time-domain DAS/MVDR or any covariance/subspace method.
#[derive(Debug, Clone)]
pub struct BeamformingLocalizationInput {
    /// Raw sensor time-series data shaped `(n_sensors, 1, n_samples)`.
    pub sensor_data: Array3<f64>,
    /// Sampling frequency in Hz (must match the acquisition used to generate `sensor_data`).
    pub sampling_frequency: f64,
}

impl BeamformingLocalizationInput {
    /// Validate invariants for the beamforming-localization input.
    pub fn validate(&self, expected_sensors: usize) -> KwaversResult<()> {
        let (n_sensors, channels, n_samples) = self.sensor_data.dim();
        if n_sensors != expected_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "BeamformingLocalizationInput: sensor_data n_sensors ({n_sensors}) does not match expected ({expected_sensors})"
            )));
        }
        if channels != 1 {
            return Err(KwaversError::InvalidInput(format!(
                "BeamformingLocalizationInput: expected channels=1, got {channels}"
            )));
        }
        if n_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "BeamformingLocalizationInput: n_samples must be > 0".to_string(),
            ));
        }
        if !self.sampling_frequency.is_finite() || self.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BeamformingLocalizationInput: sampling_frequency must be finite and > 0"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Beamforming-based localization using raw time-series data (SSOT compliant).
///
/// This is the dedicated API that allows deleting the redundant beamforming implementation
/// previously living in `sensor::localization`.
///
/// Internals:
/// - Constructs `BeamformingCoreConfig` from `LocalizationConfig`.
/// - Builds a shared `BeamformingProcessor` using sensor positions.
/// - Runs `BeamformSearch` using a caller-supplied `LocalizationBeamformSearchConfig`.
pub fn localize_beamforming(
    sensor_array: &SensorArray,
    input: &BeamformingLocalizationInput,
    search_cfg: LocalizationBeamformSearchConfig,
) -> KwaversResult<LocalizationResult> {
    use std::time::Instant;

    let start = Instant::now();

    let expected = sensor_array.num_sensors();
    input.validate(expected)?;
    search_cfg.validate()?;

    if (search_cfg.core.sampling_frequency - input.sampling_frequency).abs() > 0.0 {
        return Err(KwaversError::InvalidInput(
            "localize_beamforming: search_cfg.core.sampling_frequency must equal input.sampling_frequency to keep frequency-to-sample mapping consistent"
                .to_string(),
        ));
    }

    let sensor_positions: Vec<[f64; 3]> = sensor_array
        .get_sensor_positions()
        .iter()
        .map(|p| p.to_array())
        .collect();

    let beamformer = BeamformingProcessor::new(search_cfg.core.clone(), sensor_positions);

    let search = BeamformSearch::new(beamformer, search_cfg)?;

    let centroid = sensor_array.centroid().to_array();
    let position = search.search(centroid, &input.sensor_data)?;

    let computation_time = start.elapsed().as_secs_f64();

    Ok(LocalizationResult {
        position,
        uncertainty: Position::new(0.0, 0.0, 0.0),
        confidence: 0.0,
        method: LocalizationMethod::Beamforming,
        computation_time,
    })
}

/// Beamforming grid search evaluator for localization.
///
/// This is an orchestration component that:
/// 1. Generates (or consumes) candidate grid points.
/// 2. For each point, computes delays via the shared `BeamformingProcessor`.
/// 3. Scores the point by applying a shared beamforming method to the provided sensor data.
/// 4. Returns the point with maximum score.
///
/// It performs **no** steering-vector or covariance math beyond calls into the shared beamformer.
#[derive(Debug)]
pub struct BeamformSearch {
    processor: BeamformingProcessor,
    cfg: LocalizationBeamformSearchConfig,
}

impl BeamformSearch {
    /// Create a new beamforming grid search evaluator.
    ///
    /// # Errors
    /// Returns an error if `cfg` violates invariants.
    pub fn new(
        processor: BeamformingProcessor,
        cfg: LocalizationBeamformSearchConfig,
    ) -> KwaversResult<Self> {
        cfg.validate()?;
        Ok(Self { processor, cfg })
    }

    /// Access the underlying shared beamforming processor (SSOT).
    #[must_use]
    pub fn processor(&self) -> &BeamformingProcessor {
        &self.processor
    }

    /// Access the search configuration (policy layer).
    #[must_use]
    pub fn config(&self) -> &LocalizationBeamformSearchConfig {
        &self.cfg
    }

    /// Perform the grid search and return the best position.
    ///
    /// # Input Contract
    /// - `sensor_data` must have shape `(n_elements, 1, n_samples)`.
    /// - `n_elements` must match the processor's configured sensor count.
    ///
    /// # Errors
    /// Returns an error if input shapes are invalid or if no candidate points exist.
    pub fn search(
        &self,
        array_centroid_m: [f64; 3],
        sensor_data: &Array3<f64>,
    ) -> KwaversResult<Position> {
        let (n_elements, channels, n_samples) = sensor_data.dim();
        if channels != 1 {
            return Err(KwaversError::InvalidInput(format!(
                "BeamformSearch expects sensor_data shape (n_elements, 1, n_samples); got channels={channels}"
            )));
        }
        if n_elements != self.processor.num_sensors() {
            return Err(KwaversError::InvalidInput(format!(
                "BeamformSearch sensor_data n_elements ({n_elements}) does not match processor.num_sensors() ({})",
                self.processor.num_sensors()
            )));
        }
        if n_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "BeamformSearch requires n_samples > 0".to_string(),
            ));
        }

        let points = self.generate_points(array_centroid_m)?;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_point = None::<[f64; 3]>;

        for point in points {
            let score = self.score_point(sensor_data, point)?;
            if score > best_score {
                best_score = score;
                best_point = Some(point);
            }
        }

        let best_point = best_point.ok_or_else(|| {
            KwaversError::InvalidInput("BeamformSearch: empty candidate set".to_string())
        })?;

        Ok(Position::from_array(best_point))
    }

    fn generate_points(&self, centroid: [f64; 3]) -> KwaversResult<Vec<[f64; 3]>> {
        match &self.cfg.grid {
            SearchGrid::ExplicitPoints { points_m } => Ok(points_m.clone()),
            SearchGrid::CenteredCube {
                search_radius_m,
                grid_resolution_m,
                min_points_per_axis,
            } => {
                let r = *search_radius_m;
                let h = *grid_resolution_m;

                let span = 2.0 * r;
                let approx = (span / h).ceil() as usize + 1;
                let n = approx.max(*min_points_per_axis);

                let mut pts = Vec::with_capacity(n * n * n);
                for ix in 0..n {
                    let x = centroid[0] - r + (ix as f64) * span / ((n - 1) as f64);
                    for iy in 0..n {
                        let y = centroid[1] - r + (iy as f64) * span / ((n - 1) as f64);
                        for iz in 0..n {
                            let z = centroid[2] - r + (iz as f64) * span / ((n - 1) as f64);
                            pts.push([x, y, z]);
                        }
                    }
                }
                Ok(pts)
            }
        }
    }

    fn score_point(&self, sensor_data: &Array3<f64>, point: [f64; 3]) -> KwaversResult<f64> {
        match &self.cfg.method {
            LocalizationBeamformingMethod::SrpDasTimeDomain { delay_reference } => {
                let delays_s = self.processor.compute_delays(point);
                let weights = vec![1.0; self.processor.num_sensors()];

                let out = crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum(
                    sensor_data,
                    self.processor.config.sampling_frequency,
                    &delays_s,
                    &weights,
                    *delay_reference,
                )?;

                let (ox, oy, ot) = out.dim();
                if ox != 1 || oy != 1 {
                    return Err(KwaversError::InvalidInput(format!(
                        "BeamformSearch expected SRP-DAS output shape (1,1,n_samples); got ({ox},{oy},...)"
                    )));
                }

                let mut power = 0.0;
                for t in 0..ot {
                    let v = out[[0, 0, t]];
                    power += v * v;
                }

                if self.cfg.normalize_by_sensor_count && self.processor.num_sensors() > 0 {
                    power /= self.processor.num_sensors() as f64;
                }

                Ok(power)
            }

            LocalizationBeamformingMethod::CaponMvdrSpectrum { .. } => {
                Err(KwaversError::InvalidInput(
                    "MVDR/Capon scoring is an analysis-layer algorithm and is not available from beamforming_search::BeamformSearch. Use analysis::signal_processing::beamforming::narrowband APIs for MVDR/Capon spatial spectrum evaluation."
                        .to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_processor(n: usize) -> BeamformingProcessor {
        let cfg = BeamformingCoreConfig::default();
        let positions = (0..n).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect();
        BeamformingProcessor::new(cfg, positions)
    }

    #[test]
    fn centered_cube_generates_points() {
        let processor = make_processor(4);

        let cfg = LocalizationBeamformSearchConfig {
            method: LocalizationBeamformingMethod::SrpDasTimeDomain {
                delay_reference: DelayReference::recommended_default(),
            },
            ..Default::default()
        };

        let search = BeamformSearch::new(processor, cfg).expect("construct search");

        let pts = search
            .generate_points([0.0, 0.0, 0.0])
            .expect("generate points");
        assert!(!pts.is_empty());
    }
}
