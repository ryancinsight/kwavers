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
    /// Array of frequencies to simulate \[Hz\]
    pub frequencies: Vec<f64>,
    /// Weights for each frequency component
    pub weights: Vec<f64>,
    /// Enable harmonic generation tracking
    pub track_harmonics: bool,
    /// Maximum harmonic order to track
    pub max_harmonic_order: usize,
    /// Frequency resolution for spectral analysis \[Hz\]
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
    /// * `frequencies` - Vector of frequencies to simulate \[Hz\]
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
    /// * `fundamental` - Fundamental frequency \[Hz\]
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
    /// * `min_freq` - Minimum frequency \[Hz\]
    /// * `max_freq` - Maximum frequency \[Hz\]
    /// * `num_points` - Number of frequency points
    ///
    /// # Returns
    ///
    /// A `MultiFrequencyConfig` for broadband simulation
    #[must_use]
    pub fn broadband(min_freq: f64, max_freq: f64, num_points: usize) -> Self {
        let frequencies: Vec<f64> = Array1::linspace(min_freq, max_freq, num_points).to_vec();

        Self {
            frequencies,
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
    /// The fundamental frequency \[Hz\], or None if no frequencies are configured
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
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
    /// The bandwidth \[Hz\]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_produces_single_1mhz_component() {
        let cfg = MultiFrequencyConfig::default();
        assert_eq!(cfg.frequencies, vec![1e6]);
        assert_eq!(cfg.weights, vec![1.0]);
        assert!(!cfg.track_harmonics);
        assert_eq!(cfg.max_harmonic_order, 5);
    }

    /// `new` with `None` weights produces equal-weight components.
    #[test]
    fn new_with_none_weights_distributes_equally() {
        let freqs = vec![1e6, 2e6, 3e6];
        let cfg = MultiFrequencyConfig::new(freqs.clone(), None);
        assert_eq!(cfg.frequencies, freqs);
        assert_eq!(cfg.weights.len(), 3);
        let expected_weight = 1.0 / 3.0;
        for &w in &cfg.weights {
            assert!((w - expected_weight).abs() < 1e-15);
        }
    }

    /// `new` with explicit weights stores them verbatim.
    #[test]
    fn new_with_explicit_weights_stores_them_verbatim() {
        let freqs = vec![1e6, 2e6];
        let weights = vec![0.7, 0.3];
        let cfg = MultiFrequencyConfig::new(freqs, Some(weights.clone()));
        assert_eq!(cfg.weights, weights);
    }

    /// `for_harmonics` produces frequencies [f, 2f, …, n·f] with equal weights
    /// summing to 1 and sets `track_harmonics = true`.
    #[test]
    fn for_harmonics_produces_correct_frequency_series() {
        let fundamental = 1e6_f64;
        let n = 3usize;
        let cfg = MultiFrequencyConfig::for_harmonics(fundamental, n);
        assert_eq!(cfg.frequencies.len(), n);
        for (idx, &f) in cfg.frequencies.iter().enumerate() {
            let expected = fundamental * (idx + 1) as f64;
            assert!(
                (f - expected).abs() < 1.0,
                "harmonic {}: {f} vs {expected}",
                idx + 1
            );
        }
        assert!(cfg.track_harmonics);
        assert_eq!(cfg.max_harmonic_order, n);
        let weight_sum: f64 = cfg.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-15, "weights must sum to 1");
    }

    /// `broadband` produces endpoints matching min/max frequency and num_points entries.
    #[test]
    fn broadband_endpoints_and_count_are_correct() {
        let (min_f, max_f, n) = (1e6, 5e6, 5usize);
        let cfg = MultiFrequencyConfig::broadband(min_f, max_f, n);
        assert_eq!(cfg.frequencies.len(), n);
        assert!((cfg.frequencies[0] - min_f).abs() < 1.0);
        assert!((cfg.frequencies[n - 1] - max_f).abs() < 1.0);
        let weight_sum: f64 = cfg.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-15);
    }

    /// `validate` accepts a valid single-frequency unit-weight config.
    #[test]
    fn validate_accepts_valid_config() {
        let cfg = MultiFrequencyConfig::default();
        assert!(cfg.validate(), "default config must be valid");
    }

    /// `validate` rejects empty frequency list.
    #[test]
    fn validate_rejects_empty_frequencies() {
        let cfg = MultiFrequencyConfig {
            frequencies: vec![],
            weights: vec![],
            ..Default::default()
        };
        assert!(!cfg.validate());
    }

    /// `validate` rejects mismatched frequency / weight lengths.
    #[test]
    fn validate_rejects_mismatched_lengths() {
        let cfg = MultiFrequencyConfig {
            frequencies: vec![1e6, 2e6],
            weights: vec![1.0],
            ..Default::default()
        };
        assert!(!cfg.validate());
    }

    /// `fundamental_frequency` returns the minimum frequency.
    #[test]
    fn fundamental_frequency_returns_minimum() {
        let cfg = MultiFrequencyConfig::new(vec![3e6, 1e6, 2e6], None);
        let f0 = cfg.fundamental_frequency().unwrap();
        assert!((f0 - 1e6).abs() < 1.0);
    }

    /// `bandwidth` is max − min; single component → 0.
    #[test]
    fn bandwidth_is_max_minus_min() {
        let cfg = MultiFrequencyConfig::new(vec![1e6, 5e6], None);
        assert!((cfg.bandwidth() - 4e6).abs() < 1.0);

        let single = MultiFrequencyConfig::default();
        assert_eq!(single.bandwidth(), 0.0);
    }
}
