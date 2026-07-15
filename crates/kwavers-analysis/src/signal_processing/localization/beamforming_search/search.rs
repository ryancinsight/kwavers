use super::types::{
    BeamformingLocalizationInput, LocalizationBeamformSearchConfig, LocalizationBeamformingMethod,
    SearchGrid,
};
use crate::signal_processing::localization::LocalizationResult;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_receiver::array::{Position, SensorArray};
use kwavers_transducer::beamforming::processor::BeamformingProcessor;
use leto::Array3;

/// Beamforming-based localization using raw time-series data (SSOT compliant).
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
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
            "localize_beamforming: search_cfg.core.sampling_frequency must equal input.sampling_frequency to keep frequency-to-sample mapping consistent".to_owned(),
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
    let position_obj = search.search(centroid, &input.sensor_data)?;
    let position = position_obj.to_array();

    let _computation_time = start.elapsed().as_secs_f64();

    Ok(LocalizationResult {
        position,
        uncertainty: 0.0,
        residual: 0.0,
        iterations: 1,
        converged: true,
    })
}

/// Beamforming grid search evaluator for localization.
#[derive(Debug)]
pub struct BeamformSearch {
    pub(super) processor: BeamformingProcessor,
    pub(super) cfg: LocalizationBeamformSearchConfig,
}

impl BeamformSearch {
    /// Create a new beamforming grid search evaluator.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        processor: BeamformingProcessor,
        cfg: LocalizationBeamformSearchConfig,
    ) -> KwaversResult<Self> {
        cfg.validate()?;
        Ok(Self { processor, cfg })
    }

    /// Access the underlying shared beamforming processor (SSOT).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn processor(&self) -> &BeamformingProcessor {
        &self.processor
    }

    /// Access the search configuration (policy layer).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn config(&self) -> &LocalizationBeamformSearchConfig {
        &self.cfg
    }

    /// Perform the grid search and return the best position.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn search(
        &self,
        array_centroid_m: [f64; 3],
        sensor_data: &Array3<f64>,
    ) -> KwaversResult<Position> {
        let [n_elements, channels, n_samples] = sensor_data.shape();
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
                "BeamformSearch requires n_samples > 0".to_owned(),
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
            KwaversError::InvalidInput("BeamformSearch: empty candidate set".to_owned())
        })?;

        Ok(Position::from_array(best_point))
    }
    /// Generate points.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn generate_points(&self, centroid: [f64; 3]) -> KwaversResult<Vec<[f64; 3]>> {
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

                let out = crate::signal_processing::beamforming::time_domain::delay_and_sum(
                    sensor_data,
                    self.processor.config.sampling_frequency,
                    &delays_s,
                    &weights,
                    *delay_reference,
                )?;

                let [ox, oy, ot] = out.shape();
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
                    "MVDR/Capon scoring is an analysis-layer algorithm and is not available from beamforming_search::BeamformSearch. Use analysis::signal_processing::beamforming::narrowband APIs for MVDR/Capon spatial spectrum evaluation.".to_owned(),
                ))
            }
        }
    }
}
