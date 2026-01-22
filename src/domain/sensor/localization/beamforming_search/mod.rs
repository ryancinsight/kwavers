#![deny(missing_docs)]
//! Beamforming-based localization (grid search) built on the shared beamforming SSOT.
//!
//! # Architectural Intent (Deep Vertical, SSOT)
//! This module exists to eliminate redundant beamforming implementations inside localization.
//!
//! - **Beamforming algorithms & numerics** are owned by `crate::domain::sensor::beamforming`.
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
//! - All candidate point scoring is performed via shared `crate::domain::sensor::beamforming` SSOT APIs.
//! - MVDR covariance-domain selection is an explicit policy choice; no silent fallback is permitted.
//!
//! # MVDR covariance-domain policy (explicit)
//! For MVDR/Capon scoring, the covariance/snapshot domain is an explicit policy choice via
//! `config::MvdrCovarianceDomain` (e.g., complex baseband snapshots vs. real time-sample covariance).

pub mod config;

pub use config::{LocalizationBeamformSearchConfig, LocalizationBeamformingMethod, SearchGrid};

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::beamforming::BeamformingProcessor;
use crate::domain::sensor::localization::Position;
use ndarray::Array3;

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
        // Validate shape and consistency with processor.
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
        use config::SearchGrid;

        match &self.cfg.grid {
            SearchGrid::ExplicitPoints { points_m } => Ok(points_m.clone()),
            SearchGrid::CenteredCube {
                search_radius_m,
                grid_resolution_m,
                min_points_per_axis,
            } => {
                let r = *search_radius_m;
                let h = *grid_resolution_m;

                // Determine points per axis.
                // Invariant: min_points_per_axis >= 2 already validated.
                let span = 2.0 * r;
                let approx = (span / h).ceil() as usize + 1;
                let n = approx.max(*min_points_per_axis);

                // Build axis coordinates.
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
            config::LocalizationBeamformingMethod::SrpDasTimeDomain { delay_reference } => {
                // A) SRP-DAS (Steered Response Power, time-domain DAS)
                //
                // Field jargon:
                // - Use geometric TOF delays τ_i(p)=||x_i-p||/c.
                // - Convert τ_i into relative delays using an explicit delay datum / reference.
                // - Apply time-domain DAS and score via energy Σ_t y_p(t)^2.
                //
                // Recommended default delay datum in practice: reference sensor 0 (SensorIndex(0)),
                // but we honor the explicit policy provided by `delay_reference`.
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

            config::LocalizationBeamformingMethod::CaponMvdrSpectrum { .. } => {
                Err(KwaversError::InvalidInput(
                    "MVDR/Capon scoring is an analysis-layer algorithm and is not available from domain::sensor::localization::beamforming_search. Use analysis::signal_processing::beamforming::narrowband APIs for MVDR/Capon spatial spectrum evaluation."
                        .to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::sensor::beamforming::BeamformingCoreConfig;

    fn make_processor(n: usize) -> BeamformingProcessor {
        let cfg = BeamformingCoreConfig::default();
        let positions = (0..n).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect();
        BeamformingProcessor::new(cfg, positions)
    }

    #[test]
    fn centered_cube_generates_points() {
        use crate::analysis::signal_processing::beamforming::time_domain::DelayReference;

        let processor = make_processor(4);

        // Field-jargon correct default: SRP-DAS (time-domain DAS) with explicit delay datum.
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
