// localization/algorithms.rs - Localization algorithms

use super::{LocalizationConfig, LocalizationResult, Position, SensorArray};
use crate::error::KwaversResult;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Localization method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LocalizationMethod {
    TDOA,        // Time Difference of Arrival
    TOA,         // Time of Arrival
    Beamforming, // Beamforming-based
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
            return Err(crate::error::KwaversError::InvalidInput(
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
                return Err(crate::error::KwaversError::InvalidInput(format!(
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
            return Err(crate::error::KwaversError::InvalidInput(format!(
                "TOA requires {} measurements (one per sensor), got {}",
                array.num_sensors(),
                measurements.len()
            )));
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
        use crate::sensor::localization::{MultilaterationMethod, MultilaterationSolver};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::localization::array::{ArrayGeometry, Sensor};

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

/// Beamforming localization algorithm
struct BeamformingAlgorithm;

impl LocalizationAlgorithmImpl for BeamformingAlgorithm {
    fn localize_impl(
        &self,
        array: &SensorArray,
        measurements: &[f64],
        config: &LocalizationConfig,
    ) -> KwaversResult<LocalizationResult> {
        use std::time::Instant;
        let start = Instant::now();

        // Delay-and-sum beamforming: search grid for maximum steered response
        // Algorithm:
        // 1. Define search grid over spatial region
        // 2. For each grid point, calculate delays to all sensors
        // 3. Apply delays and sum (coherent integration)
        // 4. Find grid point with maximum output power
        //
        // References:
        // - Van Trees (2002): "Optimum Array Processing"
        // - Johnson & Dudgeon (1993): "Array Signal Processing"

        if measurements.len() != array.num_sensors() {
            return Err(crate::error::KwaversError::InvalidInput(format!(
                "Beamforming requires {} measurements (one per sensor), got {}",
                array.num_sensors(),
                measurements.len()
            )));
        }

        // Define search grid (coarse grid for efficiency)
        let search_range = config.search_radius.unwrap_or(1.0); // meters
        let grid_resolution = 0.05; // 5 cm resolution
        let n_points = ((2.0 * search_range / grid_resolution) as usize).max(10);

        let centroid = array.centroid();
        let mut best_position = centroid;
        let mut best_power = f64::NEG_INFINITY;

        // Grid search
        for i in 0..n_points {
            for j in 0..n_points {
                for k in 0..n_points {
                    let x = centroid.x - search_range
                        + (i as f64) * 2.0 * search_range / (n_points as f64);
                    let y = centroid.y - search_range
                        + (j as f64) * 2.0 * search_range / (n_points as f64);
                    let z = centroid.z - search_range
                        + (k as f64) * 2.0 * search_range / (n_points as f64);

                    let test_pos = Position::new(x, y, z);

                    // Calculate steered response power
                    let power =
                        self.calculate_steered_response(&test_pos, array, measurements, config);

                    if power > best_power {
                        best_power = power;
                        best_position = test_pos;
                    }
                }
            }
        }

        // Estimate uncertainty based on beamwidth (inversely proportional to array aperture)
        let array_size = array.max_baseline();
        let wavelength = config.sound_speed / config.frequency;
        let beamwidth = wavelength / array_size; // Approximate beamwidth in radians
        let uncertainty = beamwidth * search_range; // Spatial uncertainty

        let computation_time = start.elapsed().as_secs_f64();

        Ok(LocalizationResult {
            position: best_position,
            uncertainty: Position::new(uncertainty, uncertainty, uncertainty),
            confidence: (best_power / measurements.len() as f64).clamp(0.0, 1.0),
            method: LocalizationMethod::Beamforming,
            computation_time,
        })
    }
}

impl BeamformingAlgorithm {
    /// Calculate steered response power at a test position
    ///
    /// Implements delay-and-sum beamforming:
    /// P(r) = |∑ᵢ wᵢ xᵢ(t - τᵢ(r))|²
    /// where τᵢ(r) is the propagation delay from position r to sensor i
    fn calculate_steered_response(
        &self,
        position: &Position,
        array: &SensorArray,
        measurements: &[f64],
        config: &LocalizationConfig,
    ) -> f64 {
        let c = config.sound_speed;

        // CORRECTED: Calculate steered response with complex arithmetic
        let mut beamformed = Complex64::new(0.0, 0.0);

        for (i, &measurement) in measurements.iter().enumerate().take(array.num_sensors()) {
            let sensor_pos = array.get_sensor_position(i);
            let distance = position.distance_to(sensor_pos);
            let delay = distance / c;

            // CORRECTED: Phase shift for coherent beamforming
            // exp(j ω τ) = cos(ω τ) + j sin(ω τ)
            let phase = 2.0 * std::f64::consts::PI * config.frequency * delay;
            let steering_weight = Complex64::new(phase.cos(), phase.sin());

            // Apply steering weight and uniform weighting
            let weight = 1.0 / (array.num_sensors() as f64);
            let data_complex = Complex64::new(measurement, 0.0);
            beamformed += weight * steering_weight * data_complex;
        }

        // Return power (magnitude squared)
        beamformed.norm_sqr()
    }
}
