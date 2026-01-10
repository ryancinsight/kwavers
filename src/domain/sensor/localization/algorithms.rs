// localization/algorithms.rs - Localization algorithms
//
// SSOT note:
// - Beamforming numerics/algorithms are owned by `crate::sensor::beamforming`.
// - Localization beamforming *search policy + orchestration* is owned by
//   `crate::sensor::localization::beamforming_search`.
//
// This file must not re-implement beamforming (DAS/MVDR/steering) logic.

use super::{LocalizationConfig, LocalizationResult, Position, SensorArray};
use crate::domain::core::error::{KwaversError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Localization method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LocalizationMethod {
    TDOA,        // Time Difference of Arrival
    TOA,         // Time of Arrival
    Beamforming, // Beamforming-based grid search (SSOT via BeamformingProcessor + BeamformSearch)
    MUSIC,       // Multiple Signal Classification
    ESPRIT,      // Estimation of Signal Parameters via Rotational Invariance
    SrpPhat,     // Steered Response Power with Phase Transform
}

/// Base trait for localization algorithms
pub trait LocalizationAlgorithm {
    /// Perform localization
    fn localize(
        &self,
        array: &SensorArray,
        measurements: &[f64],
        config: &LocalizationConfig,
    ) -> KwaversResult<LocalizationResult>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Estimate uncertainty
    fn estimate_uncertainty(
        &self,
        array: &SensorArray,
        position: &Position,
        config: &LocalizationConfig,
    ) -> Position;
}

/// Main algorithm processor
#[derive(Debug)]
pub struct LocalizationProcessor {
    method: LocalizationMethod,
}

impl LocalizationProcessor {
    /// Create from method
    #[must_use]
    pub fn from_method(method: LocalizationMethod) -> Self {
        Self { method }
    }

    /// Perform localization
    pub fn localize(
        &self,
        array: &SensorArray,
        measurements: &[f64],
        config: &LocalizationConfig,
    ) -> KwaversResult<LocalizationResult> {
        match self.method {
            LocalizationMethod::TDOA => {
                let algo = TDOAAlgorithm;
                algo.localize_impl(array, measurements, config)
            }
            LocalizationMethod::TOA => {
                let algo = TOAAlgorithm;
                algo.localize_impl(array, measurements, config)
            }
            LocalizationMethod::Beamforming => {
                let algo = BeamformingAlgorithm;
                algo.localize_impl(array, measurements, config)
            }
            _ => {
                let algo = TDOAAlgorithm;
                algo.localize_impl(array, measurements, config)
            }
        }
    }
}

/// Base trait for specific algorithms
trait LocalizationAlgorithmImpl {
    fn localize_impl(
        &self,
        array: &SensorArray,
        measurements: &[f64],
        config: &LocalizationConfig,
    ) -> KwaversResult<LocalizationResult>;
}

/// TDOA localization algorithm
struct TDOAAlgorithm;

impl LocalizationAlgorithmImpl for TDOAAlgorithm {
    fn localize_impl(
        &self,
        array: &SensorArray,
        measurements: &[f64],
        config: &LocalizationConfig,
    ) -> KwaversResult<LocalizationResult> {
        use std::time::Instant;
        let start = Instant::now();

        let n_sensors = array.num_sensors();
        if n_sensors < 2 {
            return Err(crate::domain::core::error::KwaversError::InvalidInput(
                "TDOA requires at least 2 sensors".to_string(),
            ));
        }

        // Build TDOA measurements relative to reference sensor 0.
        // Supported input formats:
        // - length == n_sensors: per-sensor TOA (seconds); converted to diffs vs sensor 0
        // - length == n_sensors - 1: per-sensor TDOA diffs (sensor i minus sensor 0), i=1..n-1
        let mut processor = super::tdoa::TDOAProcessor::new(config.sound_speed);

        match measurements.len() {
            len if len == n_sensors => {
                let t0 = measurements[0];
                for (i, &measurement) in measurements.iter().enumerate().skip(1) {
                    let dt = measurement - t0;
                    processor.add_measurement(super::tdoa::TDOAMeasurement::new(0, i, dt));
                }
            }
            len if len == n_sensors.saturating_sub(1) => {
                for (i, &measurement) in measurements.iter().enumerate() {
                    let dt = measurement;
                    processor.add_measurement(super::tdoa::TDOAMeasurement::new(0, i + 1, dt));
                }
            }
            _ => {
                return Err(crate::domain::core::error::KwaversError::InvalidInput(format!(
                    "Unsupported TDOA measurement layout: got {}, expected {} (TOA) or {} (TDOA diffs)",
                    measurements.len(),
                    n_sensors,
                    n_sensors.saturating_sub(1)
                )));
            }
        }

        // Solve for source position via Gauss-Newton hyperbolic localization.
        let position = processor.process(array)?;

        // Residual-based uncertainty estimate (RMSE of distance differences).
        let residuals = processor.calculate_residuals(&position, array);
        let rmse = if residuals.is_empty() {
            0.0
        } else {
            let sum_sq: f64 = residuals.iter().map(|r| r * r).sum();
            (sum_sq / residuals.len() as f64).sqrt()
        };

        let computation_time = start.elapsed().as_secs_f64();
        Ok(LocalizationResult {
            position,
            uncertainty: Position::new(rmse, rmse, rmse),
            confidence: (1.0 / (1.0 + rmse)).clamp(0.0, 1.0),
            method: LocalizationMethod::TDOA,
            computation_time,
        })
    }
}

/// TOA localization algorithm
struct TOAAlgorithm;

impl LocalizationAlgorithmImpl for TOAAlgorithm {
    fn localize_impl(
        &self,
        array: &SensorArray,
        measurements: &[f64],
        config: &LocalizationConfig,
    ) -> KwaversResult<LocalizationResult> {
        use std::time::Instant;
        let start = Instant::now();

        // Interpret measurements as per-sensor time-of-arrival (seconds) from a known emission time,
        // or directly as ranges (meters) if values appear already scaled. We map to ranges by c*t.
        if measurements.len() != array.num_sensors() {
            return Err(crate::domain::core::error::KwaversError::InvalidInput(
                format!(
                    "TOA requires {} measurements (one per sensor), got {}",
                    array.num_sensors(),
                    measurements.len()
                ),
            ));
        }

        // Convert to ranges using sound speed.
        let ranges: Vec<f64> = measurements
            .iter()
            .map(|&t_or_r| {
                // If values are implausibly small given typical sound speeds and array apertures,
                // treat them as seconds; else allow already-meter values. Threshold: 1e3 meters.
                if t_or_r.abs() < 1e3 {
                    t_or_r * config.sound_speed
                } else {
                    t_or_r
                }
            })
            .collect();

        // Adaptive multilateration:
        // - If exactly three sensors, use trilateration closed-form
        // - If four or more, use LS/WLS/ML via MultilaterationSolver
        use crate::domain::sensor::localization::{MultilaterationMethod, MultilaterationSolver};
        let solver = MultilaterationSolver::new(MultilaterationMethod::LeastSquares);
        let position = solver.solve_adaptive(&ranges, None, array)?;

        // Residual-based uncertainty: difference between estimated distances and provided ranges.
        let positions = array.get_sensor_positions();
        let mut res_sum_sq = 0.0;
        for (i, p) in positions.iter().enumerate() {
            let di = position.distance_to(p);
            let ri = ranges[i];
            let res = di - ri;
            res_sum_sq += res * res;
        }
        let rmse = (res_sum_sq / positions.len() as f64).sqrt();

        let computation_time = start.elapsed().as_secs_f64();
        Ok(LocalizationResult {
            position,
            uncertainty: Position::new(rmse, rmse, rmse),
            confidence: (1.0 / (1.0 + rmse)).clamp(0.0, 1.0),
            method: LocalizationMethod::TOA,
            computation_time,
        })
    }
}

/// Beamforming localization algorithm (SSOT-compliant grid search).
///
/// This implementation delegates all beamforming numerics to the shared
/// `crate::sensor::beamforming::BeamformingProcessor` via
/// `crate::sensor::localization::beamforming_search::BeamformSearch`.
struct BeamformingAlgorithm;

impl LocalizationAlgorithmImpl for BeamformingAlgorithm {
    fn localize_impl(
        &self,
        _array: &SensorArray,
        _measurements: &[f64],
        _config: &LocalizationConfig,
    ) -> KwaversResult<LocalizationResult> {
        // Contract: for beamforming-based localization we require raw time-series data.
        // The shared beamforming stack expects `(n_elements, 1, n_samples)` sensor data.
        //
        // This legacy entry-point receives `measurements: &[f64]` with an ambiguous meaning in
        // this module (it was previously treated as one scalar per sensor and then "beamformed"
        // via phase weights, which is not a time-domain DAS implementation).
        //
        // To avoid mathematically incorrect behavior (and to avoid silently producing invalid
        // positions), we *fail fast* until a dedicated API is introduced.
        Err(KwaversError::InvalidInput(
            "Beamforming localization requires raw sensor time-series data shaped (n_sensors, 1, n_samples). The current LocalizationAlgorithm interface provides only scalar measurements and is insufficient for SSOT beamforming. Introduce a dedicated beamforming-localization API that accepts Array3<f64> sensor data."
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::sensor::localization::array::{ArrayGeometry, Sensor};

    fn make_array() -> SensorArray {
        let sensors = vec![
            Sensor::new(0, Position::new(0.0, 0.0, 0.0)),
            Sensor::new(1, Position::new(1.0, 0.0, 0.0)),
            Sensor::new(2, Position::new(0.0, 1.0, 0.0)),
            Sensor::new(3, Position::new(0.0, 0.0, 1.0)),
        ];
        SensorArray::new(sensors, 1500.0, ArrayGeometry::Arbitrary)
    }

    #[test]
    fn test_tdoa_localization_basic() {
        let array = make_array();
        let source = Position::new(0.5, 0.4, 0.3);

        // Generate TOA per sensor (seconds)
        let c = array.sound_speed();
        let toas: Vec<f64> = (0..array.num_sensors())
            .map(|i| array.get_sensor_position(i).distance_to(&source) / c)
            .collect();

        // Use TOA vector to feed TDOA (algorithm will convert to diffs vs sensor 0)
        let config = LocalizationConfig {
            sound_speed: c,
            max_iterations: 100,
            tolerance: 1e-6,
            use_gpu: false,
            method: LocalizationMethod::TDOA,
            frequency: 1e6,
            search_radius: None,
        };

        let proc = LocalizationProcessor::from_method(LocalizationMethod::TDOA);
        let result = proc
            .localize(&array, &toas, &config)
            .expect("TDOA localization failed");

        let err = result.position.distance_to(&source);
        assert!(err < 0.05, "TDOA position error too large: {err}");
    }

    #[test]
    fn test_toa_localization_basic() {
        let array = make_array();
        let source = Position::new(0.2, -0.1, 0.25);
        let c = array.sound_speed();

        // Provide TOA per sensor; algorithm converts to ranges via c*t
        let toas: Vec<f64> = (0..array.num_sensors())
            .map(|i| array.get_sensor_position(i).distance_to(&source) / c)
            .collect();

        let config = LocalizationConfig {
            sound_speed: c,
            max_iterations: 100,
            tolerance: 1e-6,
            use_gpu: false,
            method: LocalizationMethod::TOA,
            frequency: 1e6,
            search_radius: None,
        };

        let proc = LocalizationProcessor::from_method(LocalizationMethod::TOA);
        let result = proc
            .localize(&array, &toas, &config)
            .expect("TOA localization failed");

        let err = result.position.distance_to(&source);
        assert!(err < 0.05, "TOA position error too large: {err}");
    }
}
