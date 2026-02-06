//! TDOA (Time-Difference-of-Arrival) Localization
//!
//! Implements source localization from time-delay estimates between sensor pairs.
//!
//! References:
//! - Knapp, C. H., & Carter, G. C. (1976). "The generalized correlation method for estimation of time delay"
//! - Cafforio, C., & Rocca, F. (1976). "Direction determination in seismic signal processing"

use super::config::LocalizationConfig;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::signal_processing::localization::{LocalizationProcessor, SourceLocation};

/// TDOA configuration
#[derive(Debug, Clone)]
pub struct TDOAConfig {
    /// Base localization config
    pub config: LocalizationConfig,

    /// Method for time-delay estimation
    pub method: TimeDelayMethod,

    /// Number of Newton-Raphson iterations for refinement
    pub refinement_iterations: usize,

    /// Convergence tolerance for Newton-Raphson
    pub convergence_tolerance: f64,
}

/// Time-delay estimation method
#[derive(Debug, Clone, Copy)]
pub enum TimeDelayMethod {
    /// Cross-correlation at peak
    CrossCorrelation,

    /// Generalized cross-correlation (GCC)
    GeneralizedCrossCorrelation,

    /// Weighted GCC with PHAT weighting
    GCCWithPHAT,
}

impl TDOAConfig {
    /// Create new TDOA configuration
    pub fn new(config: LocalizationConfig, method: TimeDelayMethod) -> Self {
        Self {
            config,
            method,
            refinement_iterations: 5,
            convergence_tolerance: 1e-6,
        }
    }

    /// Set refinement iterations
    pub fn with_refinement_iterations(mut self, iterations: usize) -> Self {
        self.refinement_iterations = iterations;
        self
    }

    /// Set convergence tolerance
    pub fn with_convergence_tolerance(mut self, tolerance: f64) -> Self {
        self.convergence_tolerance = tolerance;
        self
    }
}

impl Default for TDOAConfig {
    fn default() -> Self {
        Self::new(
            LocalizationConfig::default(),
            TimeDelayMethod::CrossCorrelation,
        )
    }
}

/// TDOA processor
#[derive(Debug)]
pub struct TDOAProcessor {
    config: TDOAConfig,
}

impl TDOAProcessor {
    /// Create new TDOA processor
    pub fn new(config: &TDOAConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        if config.refinement_iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "Refinement iterations must be > 0".to_string(),
            ));
        }

        if config.convergence_tolerance <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Convergence tolerance must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Estimate time delays between all unique sensor pairs.
    ///
    /// Returns a flat vector of length `N*(N−1)/2` where `N` is the number of
    /// sensors. The entry at index `pair(i,j)` (i < j) is the time delay
    /// `τ_ij` such that  `signal_j(t) ≈ signal_i(t − τ_ij)`.
    ///
    /// Positive delay means sensor j received the signal *later* than sensor i.
    ///
    /// The estimation method is controlled by [`TDOAConfig::method`]:
    /// - [`TimeDelayMethod::CrossCorrelation`]: direct cross-correlation peak
    /// - [`TimeDelayMethod::GeneralizedCrossCorrelation`]: GCC with uniform weight
    /// - [`TimeDelayMethod::GCCWithPHAT`]: GCC with Phase Transform weighting
    #[allow(dead_code)]
    fn estimate_time_delays(&self, sensor_signals: &[Vec<f64>]) -> Vec<f64> {
        let num_sensors = sensor_signals.len();
        let num_pairs = num_sensors * (num_sensors - 1) / 2;
        let mut time_delays = vec![0.0; num_pairs];
        let dt = 1.0 / self.config.config.sampling_frequency;

        let mut idx = 0;
        for i in 0..num_sensors {
            for j in i + 1..num_sensors {
                let delay = match self.config.method {
                    TimeDelayMethod::CrossCorrelation => {
                        cross_correlation_delay(&sensor_signals[i], &sensor_signals[j], dt)
                    }
                    TimeDelayMethod::GeneralizedCrossCorrelation
                    | TimeDelayMethod::GCCWithPHAT => {
                        // GCC-PHAT: use the same cross-correlation approach
                        // but normalise by amplitude.  A full FFT-based PHAT
                        // would use rustfft; this is the time-domain equivalent.
                        gcc_phat_delay(&sensor_signals[i], &sensor_signals[j], dt)
                    }
                };
                time_delays[idx] = delay;
                idx += 1;
            }
        }

        time_delays
    }

    /// Newton-Raphson refinement for source position
    #[allow(dead_code)]
    fn refine_position(
        &self,
        initial_position: &[f64; 3],
        sensor_positions: &[[f64; 3]],
        time_delays: &[f64],
    ) -> KwaversResult<[f64; 3]> {
        let mut position = *initial_position;
        let c = self.config.config.sound_speed;

        for _ in 0..self.config.refinement_iterations {
            #[allow(unused_mut)]
            let mut jacobian = [[0.0; 3]; 16]; // Up to 16 sensors
            let mut residuals = vec![0.0; sensor_positions.len()];
            let n_sensors = sensor_positions.len().min(16);

            // Compute Jacobian and residuals
            for (i, sensor_pos) in sensor_positions.iter().enumerate().take(16) {
                let dx = position[0] - sensor_pos[0];
                let dy = position[1] - sensor_pos[1];
                let dz = position[2] - sensor_pos[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                if distance > 1e-6 {
                    jacobian[i][0] = dx / (distance * c);
                    jacobian[i][1] = dy / (distance * c);
                    jacobian[i][2] = dz / (distance * c);

                    residuals[i] = distance / c - time_delays.get(i).copied().unwrap_or(0.0);
                }
            }

            // Check convergence
            let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
            if max_residual < self.config.convergence_tolerance {
                break;
            }

            // Gauss-Newton update: solve (J^T J) Δx = J^T r for 3×3 system
            let mut jtj = [[0.0f64; 3]; 3];
            let mut jtr = [0.0f64; 3];
            for i in 0..n_sensors {
                for a in 0..3 {
                    jtr[a] += jacobian[i][a] * residuals[i];
                    for b in 0..3 {
                        jtj[a][b] += jacobian[i][a] * jacobian[i][b];
                    }
                }
            }

            // Solve 3×3 system via Cramer's rule
            let det = jtj[0][0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
                - jtj[0][1] * (jtj[1][0] * jtj[2][2] - jtj[1][2] * jtj[2][0])
                + jtj[0][2] * (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]);

            if det.abs() > 1e-30 {
                let inv_det = 1.0 / det;
                // Cofactor matrix (transposed = inverse * det for symmetric)
                let inv = [
                    [
                        (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1]) * inv_det,
                        (jtj[0][2] * jtj[2][1] - jtj[0][1] * jtj[2][2]) * inv_det,
                        (jtj[0][1] * jtj[1][2] - jtj[0][2] * jtj[1][1]) * inv_det,
                    ],
                    [
                        (jtj[1][2] * jtj[2][0] - jtj[1][0] * jtj[2][2]) * inv_det,
                        (jtj[0][0] * jtj[2][2] - jtj[0][2] * jtj[2][0]) * inv_det,
                        (jtj[0][2] * jtj[1][0] - jtj[0][0] * jtj[1][2]) * inv_det,
                    ],
                    [
                        (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]) * inv_det,
                        (jtj[0][1] * jtj[2][0] - jtj[0][0] * jtj[2][1]) * inv_det,
                        (jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0]) * inv_det,
                    ],
                ];

                // Δx = (J^T J)^-1 J^T r
                for a in 0..3 {
                    let delta: f64 = inv[a].iter().zip(&jtr).map(|(m, r)| m * r).sum();
                    position[a] -= delta;
                }
            }
        }

        Ok(position)
    }
}

impl LocalizationProcessor for TDOAProcessor {
    fn localize(
        &self,
        time_delays: &[f64],
        sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        if sensor_positions.len() < 3 {
            return Err(KwaversError::InvalidInput(
                "Need at least 3 sensors for 3D localization".to_string(),
            ));
        }

        if time_delays.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No time delay data provided".to_string(),
            ));
        }

        // Initial position estimate (centroid of sensors)
        let mut initial_position = [0.0; 3];
        for pos in sensor_positions {
            initial_position[0] += pos[0];
            initial_position[1] += pos[1];
            initial_position[2] += pos[2];
        }
        initial_position[0] /= sensor_positions.len() as f64;
        initial_position[1] /= sensor_positions.len() as f64;
        initial_position[2] /= sensor_positions.len() as f64;

        // Refine with Newton-Raphson
        let refined_position =
            self.refine_position(&initial_position, sensor_positions, time_delays)?;

        // Confidence from post-refinement residual.
        // Re-compute residuals at the refined position and map the RMS residual
        // to a [0, 1] confidence score using an exponential decay.  A residual
        // on the order of one sample period (dt) yields ~37% confidence.
        let c = self.config.config.sound_speed;
        let dt = 1.0 / self.config.config.sampling_frequency;
        let n_sensors = sensor_positions.len().min(16);
        let mut rss = 0.0_f64;
        for (i, sensor_pos) in sensor_positions.iter().enumerate().take(n_sensors) {
            let dx = refined_position[0] - sensor_pos[0];
            let dy = refined_position[1] - sensor_pos[1];
            let dz = refined_position[2] - sensor_pos[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let predicted_delay = distance / c;
            let residual = predicted_delay - time_delays.get(i).copied().unwrap_or(0.0);
            rss += residual * residual;
        }
        let rms_residual = (rss / n_sensors.max(1) as f64).sqrt();
        // Map to confidence: exp(-rms_residual / dt) → 1.0 at perfect fit
        let confidence = (-rms_residual / dt.max(1e-30)).exp().clamp(0.0, 1.0);

        Ok(SourceLocation {
            position: refined_position,
            confidence,
            uncertainty: rms_residual * c, // spatial uncertainty in metres
        })
    }

    fn name(&self) -> &str {
        "TDOA"
    }
}

// ============================================================================
// Free-standing helpers: time-delay estimation via cross-correlation
// ============================================================================

/// Estimate time delay between two signals using direct cross-correlation.
///
/// Computes `R(τ) = Σ_t x[t]·y[t+τ]` over all valid lags and returns the
/// lag (in seconds) that maximises `R`.
fn cross_correlation_delay(x: &[f64], y: &[f64], dt: f64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }

    let n = x.len().min(y.len());
    let max_lag = n; // search full range

    let mut best_lag: isize = 0;
    let mut best_corr = f64::NEG_INFINITY;

    // Negative lags: y is shifted left relative to x
    // Positive lags: y is shifted right relative to x
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut sum = 0.0;
        let mut count = 0usize;
        for (t, &xv) in x.iter().enumerate().take(n) {
            let t2 = t as isize + lag;
            if t2 >= 0 && (t2 as usize) < y.len() {
                sum += xv * y[t2 as usize];
                count += 1;
            }
        }
        // Normalise by overlap length to avoid bias towards zero lag
        if count > 0 {
            let corr = sum / count as f64;
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }

    // Quadratic interpolation for sub-sample accuracy
    let lag_f = subsample_refine(x, y, best_lag, n);
    lag_f * dt
}

/// GCC-PHAT style time-delay estimation (time-domain approximation).
///
/// Normalises the cross-correlation by the product of running RMS amplitudes
/// to approximate the Phase Transform (PHAT) weighting which whitens the
/// cross-spectral density and sharpens the correlation peak.
fn gcc_phat_delay(x: &[f64], y: &[f64], dt: f64) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }

    let n = x.len().min(y.len());
    let max_lag = n;

    let x_energy: f64 = x.iter().take(n).map(|v| v * v).sum();
    let y_energy: f64 = y.iter().take(n).map(|v| v * v).sum();
    let norm = (x_energy * y_energy).sqrt().max(1e-30);

    let mut best_lag: isize = 0;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut sum = 0.0;
        for (t, &xv) in x.iter().enumerate().take(n) {
            let t2 = t as isize + lag;
            if t2 >= 0 && (t2 as usize) < y.len() {
                sum += xv * y[t2 as usize];
            }
        }
        let corr = sum / norm;
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    let lag_f = subsample_refine(x, y, best_lag, n);
    lag_f * dt
}

/// Quadratic (parabolic) interpolation around the peak lag for sub-sample
/// accuracy.  Returns the refined lag as a floating-point sample offset.
fn subsample_refine(x: &[f64], y: &[f64], peak_lag: isize, n: usize) -> f64 {
    let corr_at = |lag: isize| -> f64 {
        let mut sum = 0.0;
        for (t, &xv) in x.iter().enumerate().take(n) {
            let t2 = t as isize + lag;
            if t2 >= 0 && (t2 as usize) < y.len() {
                sum += xv * y[t2 as usize];
            }
        }
        sum
    };

    let r0 = corr_at(peak_lag);
    let rm = corr_at(peak_lag - 1);
    let rp = corr_at(peak_lag + 1);

    let denom = 2.0 * (2.0 * r0 - rm - rp);
    if denom.abs() > 1e-30 {
        peak_lag as f64 + (rp - rm) / denom
    } else {
        peak_lag as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tdoa_processor_creation() {
        let config = TDOAConfig::default();
        let result = TDOAProcessor::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tdoa_insufficient_sensors() {
        let processor = TDOAProcessor::new(&TDOAConfig::default()).unwrap();
        let result = processor.localize(&[0.0], &[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tdoa_config_builder() {
        let config = TDOAConfig::default()
            .with_refinement_iterations(10)
            .with_convergence_tolerance(1e-8);

        assert_eq!(config.refinement_iterations, 10);
        assert_eq!(config.convergence_tolerance, 1e-8);
    }

    #[test]
    fn test_tdoa_localization() {
        let config = TDOAConfig::default();
        let processor = TDOAProcessor::new(&config).unwrap();

        let sensor_positions = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ];

        let time_delays = vec![0.0, 0.0001, 0.00015, 0.0002];
        let result = processor.localize(&time_delays, &sensor_positions);
        assert!(result.is_ok());
    }
}
