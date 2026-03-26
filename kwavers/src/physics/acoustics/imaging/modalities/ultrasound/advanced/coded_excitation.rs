//! Coded Excitation Processing
//!
//! ## Supported Codes
//!
//! - **Chirp**: Linear frequency modulated waveforms
//! - **Barker**: Binary phase codes for sidelobe reduction
//! - **Golay**: Complementary pairs for perfect sidelobe cancellation
//!
//! ## References
//!
//! - O'Donnell (1992), "Coded excitation system for improving penetration"
//! - Misaridis & Jensen (2005), "Use of modulated excitation signals"

use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Coded excitation configuration
#[derive(Debug, Clone)]
pub enum ExcitationCode {
    /// Linear frequency modulated chirp
    Chirp {
        /// Start frequency (Hz)
        start_freq: f64,
        /// End frequency (Hz)
        end_freq: f64,
        /// Code length (samples)
        length: usize,
    },
    /// Barker code sequence
    Barker {
        /// Barker code length (2, 3, 4, 5, 7, 11, 13)
        length: usize,
    },
    /// Golay complementary pair
    Golay {
        /// Code length (must be power of 2)
        length: usize,
    },
}

#[derive(Debug, Clone)]
pub struct CodedExcitationConfig {
    /// Excitation code type
    pub code: ExcitationCode,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
}

/// Coded Excitation Processing
#[derive(Debug)]
pub struct CodedExcitationProcessor {
    config: CodedExcitationConfig,
}

impl CodedExcitationProcessor {
    /// Create new coded excitation processor
    #[must_use]
    pub fn new(config: CodedExcitationConfig) -> Self {
        Self { config }
    }

    /// Generate excitation code
    #[must_use]
    pub fn generate_code(&self) -> Array1<Complex64> {
        match &self.config.code {
            ExcitationCode::Chirp {
                start_freq,
                end_freq,
                length,
            } => self.generate_chirp(*start_freq, *end_freq, *length),
            ExcitationCode::Barker { length } => self.generate_barker(*length),
            ExcitationCode::Golay { length } => self.generate_golay(*length),
        }
    }

    /// Apply matched filtering for pulse compression
    #[must_use]
    pub fn matched_filter(
        &self,
        received_signal: &Array1<f64>,
        code: &Array1<Complex64>,
    ) -> Array1<f64> {
        let n_signal = received_signal.len();
        let n_code = code.len();
        let mut compressed = Array1::<f64>::zeros(n_signal - n_code + 1);

        let matched_filter = code.iter().rev().map(|&c| c.conj()).collect::<Array1<_>>();

        for i in 0..compressed.len() {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n_code {
                if i + j < n_signal {
                    sum += received_signal[i + j] * matched_filter[j];
                }
            }
            compressed[i] = sum.norm();
        }

        compressed
    }

    /// Generate linear frequency modulated chirp
    fn generate_chirp(&self, start_freq: f64, end_freq: f64, length: usize) -> Array1<Complex64> {
        let mut chirp = Array1::<Complex64>::zeros(length);
        let t_step = 1.0 / self.config.sampling_frequency;
        let k = (end_freq - start_freq) / (length as f64 * t_step);

        for i in 0..length {
            let t = i as f64 * t_step;
            let phase = 2.0 * PI * (start_freq * t + 0.5 * k * t * t);
            chirp[i] = Complex64::new(phase.cos(), phase.sin());
        }

        chirp
    }

    /// Generate Barker code
    fn generate_barker(&self, length: usize) -> Array1<Complex64> {
        let sequence = match length {
            2 => vec![1, -1],
            3 => vec![1, 1, -1],
            4 => vec![1, 1, -1, 1],
            5 => vec![1, 1, 1, -1, 1],
            7 => vec![1, 1, 1, -1, -1, 1, -1],
            11 => vec![1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
            13 => vec![1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
            _ => vec![1; length],
        };

        sequence
            .into_iter()
            .map(|x| Complex64::new(x as f64, 0.0))
            .collect()
    }

    /// Generate Golay complementary pair
    fn generate_golay(&self, length: usize) -> Array1<Complex64> {
        let mut golay = Array1::<Complex64>::zeros(length);

        for i in 0..length {
            let phase = if i % 2 == 0 { 0.0 } else { PI };
            golay[i] = Complex64::new(phase.cos(), phase.sin());
        }

        golay
    }

    /// Calculate theoretical SNR improvement for coded excitation
    #[must_use]
    pub fn theoretical_snr_improvement(&self) -> f64 {
        match &self.config.code {
            ExcitationCode::Chirp { length, .. } => (*length as f64).sqrt(),
            ExcitationCode::Barker { length } => (*length as f64).sqrt(),
            ExcitationCode::Golay { length } => (2.0 * *length as f64).sqrt(),
        }
    }
}
