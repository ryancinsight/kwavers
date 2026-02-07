//! Spectral Analysis for Doppler Signals

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array1;

/// Spectral analysis configuration
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    pub fft_size: usize,
    pub overlap: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            fft_size: 256,
            overlap: 0.75,
        }
    }
}

/// Spectral analysis processor
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    #[allow(dead_code)] // Used when FFT-based PSD is implemented
    config: SpectralConfig,
}

impl SpectralAnalysis {
    pub fn new(config: SpectralConfig) -> Self {
        Self { config }
    }

    /// Compute power spectral density
    ///
    /// # Errors
    /// Returns `KwaversError::NotImplemented` — FFT-based spectral estimation pending.
    pub fn compute_psd(&self, _n_samples: usize) -> KwaversResult<Array1<f64>> {
        Err(KwaversError::NotImplemented(
            "Spectral Doppler PSD computation not yet implemented. \
             Requires Welch’s method with configurable FFT size, \
             windowing, and overlap."
                .into(),
        ))
    }
}
