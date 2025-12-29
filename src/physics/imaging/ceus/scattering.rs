//! Nonlinear Scattering for Contrast-Enhanced Ultrasound
//!
//! Implements harmonic generation, subharmonic emission, and nonlinear
//! acoustic scattering from microbubble contrast agents.

use crate::error::KwaversResult;
use crate::physics::imaging::ceus::MicrobubblePopulation;

/// Nonlinear scattering model for microbubbles
#[derive(Debug)]
pub struct NonlinearScattering {
    /// Harmonic generation efficiency
    harmonic_efficiency: f64,
}

impl NonlinearScattering {
    /// Create new nonlinear scattering model
    pub fn new() -> KwaversResult<Self> {
        Ok(Self {
            harmonic_efficiency: 0.3,
        })
    }

    /// Compute nonlinear scattering response
    ///
    /// # Arguments
    ///
    /// * `population` - Microbubble population
    /// * `concentration` - Local bubble concentration
    /// * `acoustic_pressure` - Incident acoustic pressure (Pa)
    /// * `frequency` - Acoustic frequency (Hz)
    ///
    /// # Returns
    ///
    /// Nonlinear scattering coefficient
    pub fn compute_scattering(
        &self,
        population: &MicrobubblePopulation,
        concentration: f64,
        acoustic_pressure: f64,
        frequency: f64,
    ) -> KwaversResult<f64> {
        if concentration <= 0.0 {
            return Ok(0.0);
        }

        // Base linear scattering
        let linear_scattering = population.effective_scattering(frequency);

        // Nonlinear enhancement based on acoustic pressure
        let pressure_factor = (acoustic_pressure / 100_000.0).min(1.0); // Normalize to 100 kPa

        // Resonance enhancement
        let resonance_freq = population
            .reference_bubble
            .resonance_frequency(101325.0, 1000.0);
        let freq_ratio = frequency / resonance_freq;
        let resonance_factor = 1.0 / (1.0 + (freq_ratio - 1.0).powi(2));

        // Total nonlinear scattering
        let nonlinear_scattering = linear_scattering
            * concentration
            * (1.0 + self.harmonic_efficiency * pressure_factor)
            * resonance_factor;

        Ok(nonlinear_scattering)
    }
}

/// Harmonic imaging for CEUS
#[derive(Debug)]
pub struct HarmonicImaging {
    /// Harmonic frequencies to extract
    harmonic_frequencies: Vec<f64>,
    /// Imaging parameters
    pub parameters: HarmonicImagingParameters,
}

#[derive(Debug, Clone)]
pub struct HarmonicImagingParameters {
    /// Transmit frequency (Hz)
    pub transmit_freq: f64,
    /// Receive bandwidth (Hz)
    pub bandwidth: f64,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Frame rate (Hz)
    pub frame_rate: f64,
}

impl HarmonicImaging {
    /// Create new harmonic imaging system
    pub fn new(fundamental_freq: f64) -> Self {
        let harmonic_frequencies = vec![
            fundamental_freq * 2.0, // Second harmonic
            fundamental_freq * 1.5, // Ultraharmonic
            fundamental_freq * 0.5, // Subharmonic
        ];

        Self {
            harmonic_frequencies,
            parameters: HarmonicImagingParameters {
                transmit_freq: fundamental_freq,
                bandwidth: fundamental_freq * 0.5,
                mechanical_index: 0.1,
                frame_rate: 15.0,
            },
        }
    }

    /// Extract harmonic components from scattered signal
    pub fn extract_harmonics(&self, signal: &[f64], sample_rate: f64) -> Vec<f64> {
        self.harmonic_frequencies
            .iter()
            .map(|&freq| self.extract_single_frequency(signal, freq, sample_rate))
            .collect()
    }

    /// Extract single frequency component using DFT
    fn extract_single_frequency(&self, signal: &[f64], frequency: f64, sample_rate: f64) -> f64 {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }

        let mut real = 0.0;
        let mut imag = 0.0;

        for (i, &sample) in signal.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate;
            real += sample * phase.cos();
            imag += sample * phase.sin();
        }

        (real * real + imag * imag).sqrt() / n as f64
    }
}
