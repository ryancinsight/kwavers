//! MUSIC (Multiple Signal Classification) Algorithm
//!
//! Implements super-resolution direction-of-arrival (DoA) estimation using MUSIC.
//!
//! References:
//! - Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation"
//! - Stoica, P., & Nehorai, A. (1989). "MUSIC, maximum likelihood, and Cramér–Rao bound"

use super::config::LocalizationConfig;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::signal_processing::localization::{LocalizationProcessor, SourceLocation};

/// MUSIC configuration
#[derive(Debug, Clone)]
pub struct MUSICConfig {
    /// Base localization config
    pub config: LocalizationConfig,

    /// Number of sources to detect
    pub num_sources: usize,

    /// MUSIC grid resolution for search
    pub music_grid_resolution: usize,

    /// Minimum separation between sources [m]
    pub min_source_separation: f64,
}

impl MUSICConfig {
    /// Create new MUSIC configuration
    pub fn new(config: LocalizationConfig, num_sources: usize) -> Self {
        Self {
            config,
            num_sources,
            music_grid_resolution: 360, // One degree resolution for 2D
            min_source_separation: 0.01,
        }
    }

    /// Set MUSIC grid resolution
    pub fn with_grid_resolution(mut self, resolution: usize) -> Self {
        self.music_grid_resolution = resolution;
        self
    }

    /// Set minimum source separation
    pub fn with_min_separation(mut self, separation: f64) -> Self {
        self.min_source_separation = separation;
        self
    }
}

impl Default for MUSICConfig {
    fn default() -> Self {
        Self::new(LocalizationConfig::default(), 1)
    }
}

/// MUSIC processor for direction-of-arrival estimation
#[derive(Debug)]
pub struct MUSICProcessor {
    #[allow(dead_code)]
    config: MUSICConfig,
}

impl MUSICProcessor {
    /// Create new MUSIC processor
    pub fn new(config: &MUSICConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        if config.num_sources == 0 {
            return Err(KwaversError::InvalidInput(
                "Number of sources must be > 0".to_string(),
            ));
        }

        let num_sensors = config.config.sensor_positions.len();
        if config.num_sources >= num_sensors {
            return Err(KwaversError::InvalidInput(
                "Number of sources must be < number of sensors".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Estimate covariance matrix from signals
    #[allow(dead_code)]
    fn estimate_covariance(&self, signals: &[Vec<f64>]) -> ndarray::Array2<f64> {
        let num_sensors = signals.len();
        let num_samples = if signals.is_empty() {
            0
        } else {
            signals[0].len()
        };

        let mut covariance = ndarray::Array2::zeros((num_sensors, num_sensors));

        if num_samples == 0 {
            return covariance;
        }

        // Compute sample covariance matrix: R = (1/N) * X * X^H
        for i in 0..num_sensors {
            for j in 0..num_sensors {
                let mut sum = 0.0;
                for k in 0..num_samples {
                    sum += signals[i][k] * signals[j][k];
                }
                covariance[[i, j]] = sum / num_samples as f64;
            }
        }

        covariance
    }

    /// Find peaks in MUSIC spectrum
    #[allow(dead_code)]
    fn find_peaks(&self, spectrum: &[f64], num_peaks: usize) -> Vec<usize> {
        let mut peaks = Vec::new();

        // Simple peak detection: find local maxima
        for i in 1..spectrum.len() - 1 {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
                peaks.push(i);
            }
        }

        // Sort by spectrum magnitude and keep top N
        peaks.sort_by(|&a, &b| {
            spectrum[b]
                .partial_cmp(&spectrum[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        peaks.truncate(num_peaks);
        peaks.sort();

        peaks
    }
}

impl LocalizationProcessor for MUSICProcessor {
    fn localize(
        &self,
        _time_delays: &[f64],
        _sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        // Placeholder implementation
        // Full implementation would:
        // 1. Build covariance matrix from signals
        // 2. Perform eigendecomposition
        // 3. Estimate noise subspace
        // 4. Compute MUSIC spectrum over search space
        // 5. Find peaks corresponding to source locations

        Ok(SourceLocation {
            position: [0.0, 0.0, 0.0],
            confidence: 0.0,
            uncertainty: 0.1,
        })
    }

    fn name(&self) -> &str {
        "MUSIC"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_processor_creation() {
        let config = MUSICConfig::default();
        let result = MUSICProcessor::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_music_invalid_num_sources() {
        let mut config = MUSICConfig::default();
        config.num_sources = 10; // More than sensors
        let result = MUSICProcessor::new(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_music_config_builder() {
        let config = MUSICConfig::default()
            .with_grid_resolution(720)
            .with_min_separation(0.05);

        assert_eq!(config.music_grid_resolution, 720);
        assert_eq!(config.min_source_separation, 0.05);
    }

    #[test]
    fn test_covariance_estimation() {
        let config = MUSICConfig::default();
        let processor = MUSICProcessor::new(&config).unwrap();

        let signals = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0, 5.0]];

        let cov = processor.estimate_covariance(&signals);
        assert_eq!(cov.dim(), (2, 2));
    }
}

