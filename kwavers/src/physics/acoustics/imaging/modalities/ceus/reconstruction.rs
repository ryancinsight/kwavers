//! CEUS Image Reconstruction - Contrast-Enhanced Ultrasound Imaging
//!
//! ## Mathematical Theorems and Foundations
//!
//! ### Harmonic Imaging Theorem
//! **Theorem**: Microbubble oscillations generate nonlinear harmonics at integer multiples of fundamental frequency
//! **Foundation**: Nonlinear bubble dynamics produce 2nd, 3rd, and ultraharmonics (de Jong et al. 2002)
//! **Mathematical Basis**: Rayleigh-Plesset equation with nonlinear terms predicts harmonic generation
//!
//! ### Pulse Inversion Theorem
//! **Theorem**: Phase-inverted pulses cancel linear tissue signals while reinforcing nonlinear contrast signals
//! **Foundation**: Tissue responds linearly, microbubbles respond nonlinearly (Simpson et al. 1999)
//! **Mathematical Basis**: f(x) - f(-x) = 0 for linear systems, ≠0 for nonlinear systems
//!
//! ### Amplitude Modulation Theorem
//! **Theorem**: Dual-frequency excitation enhances contrast-to-tissue ratio through destructive interference
//! **Foundation**: Tissue and contrast signals have different frequency responses (Phillips 2001)
//! **Mathematical Basis**: Complex envelope demodulation separates fundamental and harmonic components
//!
//! ### Microbubble Scattering Cross-Section
//! **Theorem**: σ ∝ (ω²R₀³)² for Rayleigh scattering regime, σ ∝ ω⁴R₀⁶ for Mie scattering
//! **Foundation**: Acoustic scattering theory applied to compressible spheres (Anderson 1950)
//! **Mathematical Basis**: Solution to Helmholtz equation with radiation boundary conditions
//!
//! ## Literature References
//! - de Jong, N. et al. (2002): "Principles and recent developments in ultrasound contrast agents"
//! - Simpson, D.H. et al. (1999): "Pulse inversion Doppler: A new method for detecting nonlinear echoes"
//! - Phillips, P. (2001): "Contrast pulse sequences (CPS): Imaging nonlinear microbubbles"
//! - Anderson, V.C. (1950): "Sound scattering from a fluid sphere"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::ceus::CEUSImagingParameters;
use ndarray::Array3;

/// CEUS image reconstruction
#[derive(Debug)]
pub struct CEUSReconstruction {
    /// Imaging parameters
    pub parameters: CEUSImagingParameters,
    /// Harmonic filter bank
    harmonic_filters: Vec<HarmonicFilter>,
}

impl CEUSReconstruction {
    /// Create new CEUS reconstruction
    pub fn new(_grid: &Grid) -> KwaversResult<Self> {
        let parameters = CEUSImagingParameters::default();

        // Create harmonic filters for different frequencies
        let harmonic_filters = vec![
            HarmonicFilter::new(parameters.frequency * 2.0, 0.1), // 2nd harmonic
            HarmonicFilter::new(parameters.frequency * 1.5, 0.1), // Ultraharmonic
        ];

        Ok(Self {
            parameters,
            harmonic_filters,
        })
    }

    /// Process frame of scattered signals into contrast image
    pub fn process_frame(&self, scattered_signals: &Array3<f64>) -> KwaversResult<ContrastImage> {
        // Apply nonlinear beamforming
        let beamformed = self.nonlinear_beamforming(scattered_signals)?;

        // Extract harmonic components
        let harmonic_image = self.extract_harmonics(&beamformed)?;

        // Apply contrast-specific processing
        let contrast_image = self.contrast_enhancement(&harmonic_image)?;

        Ok(ContrastImage {
            intensity: contrast_image,
        })
    }

    /// Nonlinear beamforming for contrast signals
    fn nonlinear_beamforming(&self, signals: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Simplified beamforming - in practice this would be much more complex
        // involving delay-and-sum with apodization, etc.

        let mut beamformed = signals.clone();

        // Apply coherence weighting for microbubble signal enhancement
        let (nx, ny, nz) = beamformed.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Simple amplitude weighting based on signal strength
                    let signal_strength = signals[[i, j, k]].abs();
                    if signal_strength > 0.0 {
                        beamformed[[i, j, k]] *= (signal_strength / 1e6).min(1.0);
                    }
                }
            }
        }

        Ok(beamformed)
    }

    /// Extract harmonic components
    fn extract_harmonics(&self, beamformed: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut harmonic_image = Array3::zeros(beamformed.raw_dim());
        let (nx, ny, nz) = beamformed.dim();

        // Apply harmonic filters
        for filter in &self.harmonic_filters {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let signal = beamformed[[i, j, k]];
                        let filtered = filter.apply(signal);
                        harmonic_image[[i, j, k]] += filtered;
                    }
                }
            }
        }

        Ok(harmonic_image)
    }

    /// Apply contrast enhancement processing
    fn contrast_enhancement(&self, harmonic_image: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut enhanced = harmonic_image.clone();

        // Apply log compression
        let (nx, ny, nz) = enhanced.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let signal = enhanced[[i, j, k]].abs(); // Ensure positive for log
                    if signal > 1e-12 {
                        // Very small threshold to avoid log(0)
                        // Convert to dB scale
                        enhanced[[i, j, k]] = 20.0 * (signal / 1e-6).log10();
                    } else {
                        enhanced[[i, j, k]] = -60.0; // Noise floor
                    }
                }
            }
        }

        // Apply dynamic range compression
        let max_intensity = enhanced.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_intensity = enhanced.iter().cloned().fold(f64::INFINITY, f64::min);

        if max_intensity > min_intensity {
            for intensity in enhanced.iter_mut() {
                let normalized = (*intensity - min_intensity) / (max_intensity - min_intensity);
                *intensity = (normalized * 255.0).clamp(0.0, 255.0);
            }
        } else {
            // If all values are the same, set to mid-range
            for intensity in enhanced.iter_mut() {
                *intensity = 127.5;
            }
        }

        Ok(enhanced)
    }
}

/// Contrast-enhanced ultrasound image
#[derive(Debug, Clone)]
pub struct ContrastImage {
    /// Intensity values (dB or normalized)
    pub intensity: Array3<f64>,
}

impl ContrastImage {
    /// Get image statistics
    #[must_use]
    pub fn statistics(&self) -> ImageStatistics {
        let values: Vec<f64> = self.intensity.iter().cloned().collect();

        if values.is_empty() {
            return ImageStatistics::default();
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        ImageStatistics {
            mean,
            std_dev,
            min: min_val,
            max: max_val,
            dynamic_range: max_val - min_val,
        }
    }
}

/// Image statistics
#[derive(Debug, Clone)]
pub struct ImageStatistics {
    /// Mean intensity
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum intensity
    pub min: f64,
    /// Maximum intensity
    pub max: f64,
    /// Dynamic range
    pub dynamic_range: f64,
}

impl Default for ImageStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            dynamic_range: 0.0,
        }
    }
}

/// Harmonic filter for frequency-specific processing
#[derive(Debug)]
struct HarmonicFilter {
    /// Filter coefficients
    coefficients: Vec<f64>,
}

impl HarmonicFilter {
    /// Create new harmonic filter
    fn new(_center_freq: f64, _bandwidth: f64) -> Self {
        // Bandpass filter for harmonic microbubble signals
        let coefficients = vec![1.0, 0.5, 0.25]; // Example FIR coefficients

        Self { coefficients }
    }

    /// Apply filter to signal
    fn apply(&self, signal: f64) -> f64 {
        // Simplified filtering - in practice would use proper IIR/FIR filtering
        signal * self.coefficients[0] // Just pass through for now
    }
}
