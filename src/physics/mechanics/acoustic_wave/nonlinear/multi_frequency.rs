//! Multi-frequency simulation configuration
//!
//! This module provides configuration and utilities for multi-frequency acoustic simulations.

use ndarray::Array1;

/// Configuration for multi-frequency acoustic simulations.
///
/// Enables analysis at multiple frequencies simultaneously, useful for:
/// - Harmonic generation studies
/// - Broadband acoustic simulations
/// - Frequency-dependent absorption modeling
#[derive(Debug, Clone)]
pub struct MultiFrequencyConfig {
    /// Array of frequencies to simulate [Hz]
    pub frequencies: Vec<f64>,
    /// Weights for each frequency component
    pub weights: Vec<f64>,
    /// Enable harmonic generation tracking
    pub track_harmonics: bool,
    /// Maximum harmonic order to track
    pub max_harmonic_order: usize,
    /// Frequency resolution for spectral analysis [Hz]
    pub frequency_resolution: f64,
}

impl Default for MultiFrequencyConfig {
    fn default() -> Self {
        Self {
            frequencies: vec![1e6], // Default 1 MHz
            weights: vec![1.0],
            track_harmonics: false,
            max_harmonic_order: 5,
            frequency_resolution: 1e3, // 1 kHz resolution
        }
    }
}

impl MultiFrequencyConfig {
    /// Creates a new multi-frequency configuration.
    ///
    /// # Arguments
    ///
    /// * `frequencies` - Vector of frequencies to simulate [Hz]
    /// * `weights` - Optional weights for each frequency (defaults to equal weights)
    ///
    /// # Returns
    ///
    /// A new `MultiFrequencyConfig` instance
    #[must_use]
    pub fn new(frequencies: Vec<f64>, weights: Option<Vec<f64>>) -> Self {
        let weights =
            weights.unwrap_or_else(|| vec![1.0 / frequencies.len() as f64; frequencies.len()]);

        Self {
            frequencies,
            weights,
            ..Default::default()
        }
    }

    /// Creates a configuration for harmonic analysis.
    ///
    /// # Arguments
    ///
    /// * `fundamental` - Fundamental frequency [Hz]
    /// * `num_harmonics` - Number of harmonics to track
    ///
    /// # Returns
    ///
    /// A `MultiFrequencyConfig` configured for harmonic analysis
    #[must_use]
    pub fn for_harmonics(fundamental: f64, num_harmonics: usize) -> Self {
        let frequencies: Vec<f64> = (1..=num_harmonics)
            .map(|n| fundamental * n as f64)
            .collect();

        Self {
            frequencies,
            weights: vec![1.0 / num_harmonics as f64; num_harmonics],
            track_harmonics: true,
            max_harmonic_order: num_harmonics,
            frequency_resolution: fundamental / 10.0,
        }
    }

    /// Creates a configuration for broadband simulation.
    ///
    /// # Arguments
    ///
    /// * `min_freq` - Minimum frequency [Hz]
    /// * `max_freq` - Maximum frequency [Hz]
    /// * `num_points` - Number of frequency points
    ///
    /// # Returns
    ///
    /// A `MultiFrequencyConfig` for broadband simulation
    #[must_use]
    pub fn broadband(min_freq: f64, max_freq: f64, num_points: usize) -> Self {
        let frequencies: Vec<f64> = Array1::linspace(min_freq, max_freq, num_points).to_vec();

        Self {
            frequencies: frequencies.clone(),
            weights: vec![1.0 / num_points as f64; num_points],
            track_harmonics: false,
            max_harmonic_order: 0,
            frequency_resolution: (max_freq - min_freq) / (num_points - 1) as f64,
        }
    }

    /// Validates the configuration.
    ///
    /// # Returns
    ///
    /// `true` if the configuration is valid, `false` otherwise
    #[must_use]
    pub fn validate(&self) -> bool {
        !self.frequencies.is_empty()
            && self.frequencies.len() == self.weights.len()
            && self.frequencies.iter().all(|&f| f > 0.0)
            && self.weights.iter().all(|&w| w >= 0.0)
            && (self.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6
    }

    /// Gets the fundamental frequency (lowest frequency).
    ///
    /// # Returns
    ///
    /// The fundamental frequency [Hz], or None if no frequencies are configured
    #[must_use]
    pub fn fundamental_frequency(&self) -> Option<f64> {
        self.frequencies
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }

    /// Gets the bandwidth of the configured frequencies.
    ///
    /// # Returns
    ///
    /// The bandwidth [Hz]
    #[must_use]
    pub fn bandwidth(&self) -> f64 {
        if self.frequencies.is_empty() {
            0.0
        } else {
            let min = self
                .frequencies
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max = self
                .frequencies
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            max - min
        }
    }
}
