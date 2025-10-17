//! Frequency Response Module
//!
//! Models the frequency-dependent behavior of transducers including
//! bandwidth, sensitivity, and impedance characteristics.

use crate::error::{ConfigError, KwaversError, KwaversResult};
use ndarray::Array1;
use num_complex::Complex64;

/// Frequency response characteristics of a transducer
///
/// Based on KLM model and Mason equivalent circuit
/// References:
/// - Krimholtz et al. (1970): "New equivalent circuits for elementary piezoelectric transducers"
/// - Mason (1948): "Electromechanical Transducers and Wave Filters"
#[derive(Debug, Clone)]
pub struct FrequencyResponse {
    /// Center frequency (Hz)
    pub center_frequency: f64,
    /// -3 dB bandwidth (Hz)
    pub bandwidth_3db: f64,
    /// -6 dB bandwidth (Hz)
    pub bandwidth_6db: f64,
    /// Fractional bandwidth (%)
    pub fractional_bandwidth: f64,
    /// Quality factor
    pub quality_factor: f64,
    /// Frequency vector (Hz)
    pub frequencies: Array1<f64>,
    /// Magnitude response (normalized)
    pub magnitude: Array1<f64>,
    /// Phase response (radians)
    pub phase: Array1<f64>,
    /// Electrical impedance (complex)
    pub impedance: Array1<Complex64>,
}

impl FrequencyResponse {
    /// Calculate frequency response using KLM model
    ///
    /// # Arguments
    /// * `center_freq` - Center frequency in Hz
    /// * `coupling` - Electromechanical coupling coefficient
    /// * `mechanical_q` - Mechanical quality factor
    /// * `electrical_q` - Electrical quality factor
    /// * `num_points` - Number of frequency points
    pub fn from_klm_model(
        center_freq: f64,
        coupling: f64,
        mechanical_q: f64,
        electrical_q: f64,
        num_points: usize,
    ) -> KwaversResult<Self> {
        if center_freq <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "center_frequency".to_string(),
                value: center_freq.to_string(),
                constraint: "Center frequency must be positive".to_string(),
            }));
        }

        // Create frequency vector (0.5 to 1.5 times center frequency)
        let freq_min = 0.5 * center_freq;
        let freq_max = 1.5 * center_freq;
        let frequencies = Array1::linspace(freq_min, freq_max, num_points);

        // Calculate normalized frequency
        let mut magnitude = Array1::zeros(num_points);
        let mut phase = Array1::zeros(num_points);
        let mut impedance = Array1::zeros(num_points);

        // Effective bandwidth factor
        let _bandwidth_factor = coupling.powi(2) / mechanical_q.sqrt();

        for (i, &freq) in frequencies.iter().enumerate() {
            let normalized_freq = freq / center_freq;
            let delta = normalized_freq - 1.0;

            // Mason model response
            let mechanical_term = Complex64::new(1.0, 2.0 * delta * mechanical_q);
            let electrical_term = Complex64::new(1.0, 2.0 * delta * electrical_q);

            // Combined response
            let response = coupling.powi(2) / (mechanical_term * electrical_term);

            magnitude[i] = response.norm();
            phase[i] = response.arg();

            // Electrical impedance: Z = Z₀ × electrical transfer function
            // Per Kinsler et al. (2000) "Fundamentals of Acoustics" Ch. 10
            // Nominal 50Ω reference impedance standard for RF systems
            let z0 = 50.0; // Nominal impedance
            impedance[i] = z0 * electrical_term;
        }

        // Normalize magnitude
        let max_mag = magnitude.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_mag > 0.0 {
            magnitude /= max_mag;
        }

        // Find bandwidth points
        let (bandwidth_3db, bandwidth_6db) = Self::find_bandwidth_points(&frequencies, &magnitude);
        let fractional_bandwidth = 100.0 * bandwidth_3db / center_freq;
        let quality_factor = center_freq / bandwidth_3db;

        Ok(Self {
            center_frequency: center_freq,
            bandwidth_3db,
            bandwidth_6db,
            fractional_bandwidth,
            quality_factor,
            frequencies,
            magnitude,
            phase,
            impedance,
        })
    }

    /// Find -3dB and -6dB bandwidth points
    fn find_bandwidth_points(frequencies: &Array1<f64>, magnitude: &Array1<f64>) -> (f64, f64) {
        let threshold_3db = 1.0 / 2.0_f64.sqrt(); // -3 dB
        let threshold_6db = 0.5; // -6 dB

        let mut freq_low_3db = frequencies[0];
        let mut freq_high_3db = frequencies[frequencies.len() - 1];
        let mut freq_low_6db = frequencies[0];
        let mut freq_high_6db = frequencies[frequencies.len() - 1];

        // Find max index
        let max_idx = magnitude
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        // Search for -3dB points
        for i in (0..max_idx).rev() {
            if magnitude[i] < threshold_3db {
                freq_low_3db = frequencies[i];
                break;
            }
        }

        for i in max_idx..frequencies.len() {
            if magnitude[i] < threshold_3db {
                freq_high_3db = frequencies[i];
                break;
            }
        }

        // Search for -6dB points
        for i in (0..max_idx).rev() {
            if magnitude[i] < threshold_6db {
                freq_low_6db = frequencies[i];
                break;
            }
        }

        for i in max_idx..frequencies.len() {
            if magnitude[i] < threshold_6db {
                freq_high_6db = frequencies[i];
                break;
            }
        }

        let bandwidth_3db = freq_high_3db - freq_low_3db;
        let bandwidth_6db = freq_high_6db - freq_low_6db;

        (bandwidth_3db, bandwidth_6db)
    }

    /// Calculate pulse response characteristics
    ///
    /// Returns (pulse length, axial resolution) in meters
    #[must_use]
    pub fn pulse_characteristics(&self, sound_speed: f64) -> (f64, f64) {
        // Pulse length ≈ (cycles * wavelength)
        // For typical transducers: 2-3 cycles
        let cycles = 2.5;
        let wavelength = sound_speed / self.center_frequency;
        let pulse_length = cycles * wavelength;

        // Axial resolution ≈ pulse_length / 2
        let axial_resolution = pulse_length / 2.0;

        (pulse_length, axial_resolution)
    }

    /// Calculate sensitivity roll-off at a given frequency
    #[must_use]
    pub fn sensitivity_at_frequency(&self, frequency: f64) -> f64 {
        // Linear interpolation in the magnitude response
        let idx = self
            .frequencies
            .iter()
            .position(|&f| f >= frequency)
            .unwrap_or(self.frequencies.len() - 1);

        if idx == 0 {
            self.magnitude[0]
        } else if idx >= self.frequencies.len() {
            self.magnitude[self.frequencies.len() - 1]
        } else {
            // Linear interpolation
            let f1 = self.frequencies[idx - 1];
            let f2 = self.frequencies[idx];
            let m1 = self.magnitude[idx - 1];
            let m2 = self.magnitude[idx];

            let t = (frequency - f1) / (f2 - f1);
            m1 * (1.0 - t) + m2 * t
        }
    }

    /// Check if response meets bandwidth requirements
    #[must_use]
    pub fn validate_bandwidth(&self, min_fractional_bw: f64) -> bool {
        self.fractional_bandwidth >= min_fractional_bw
    }

    /// Calculate insertion loss at center frequency
    #[must_use]
    pub fn insertion_loss(&self) -> f64 {
        // Insertion loss from impedance mismatch: IL = -10log₁₀(1-|Γ|²)
        // Reflection coefficient Γ = (Z-Z₀)/(Z+Z₀)
        // Per IEEE Std 177: "Standard Definitions and Methods of Measurement"
        let z0 = 50.0; // Reference impedance
        let center_idx = self.frequencies.len() / 2;
        let z = self.impedance[center_idx];

        let reflection_coeff = (z - z0) / (z + z0);
        let transmission = 1.0 - reflection_coeff.norm().powi(2);

        -10.0 * transmission.log10()
    }
}
