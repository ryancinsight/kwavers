//! Nonlinear Scattering for Contrast-Enhanced Ultrasound
//!
//! Implements harmonic generation, subharmonic emission, and nonlinear
//! acoustic scattering from microbubble contrast agents.

use super::microbubble::MicrobubblePopulation;
use aequitas::systems::si::quantities::{
    DynamicViscosity, Frequency, MassDensity, NumberDensity, Pressure, ReciprocalLength, Velocity,
};
use kwavers_core::constants::cavitation::VISCOSITY_WATER;
use kwavers_core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;

/// Nonlinear scattering model for microbubbles
#[derive(Debug)]
pub struct NonlinearScattering {
    /// Harmonic generation efficiency
    harmonic_efficiency: f64,
}

impl NonlinearScattering {
    /// Create new nonlinear scattering model
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn compute_scattering(
        &self,
        population: &MicrobubblePopulation,
        concentration: NumberDensity<f64>,
        acoustic_pressure: Pressure<f64>,
        frequency: Frequency<f64>,
    ) -> KwaversResult<ReciprocalLength<f64>> {
        if concentration.into_base() <= 0.0 {
            return Ok(ReciprocalLength::from_base(0.0));
        }

        let linear_scattering = population
            .effective_scattering(
                frequency,
                Pressure::from_base(ATMOSPHERIC_PRESSURE),
                MassDensity::from_base(DENSITY_WATER_NOMINAL),
                Velocity::from_base(SOUND_SPEED_WATER_SIM),
                DynamicViscosity::from_base(VISCOSITY_WATER),
            )?
            .into_base();

        // Nonlinear enhancement based on acoustic pressure
        let pressure_factor = (acoustic_pressure.into_base() / 100_000.0).min(1.0); // Normalize to 100 kPa

        // Resonance enhancement
        let resonance_freq = population
            .reference_bubble
            .resonance_frequency(
                Pressure::from_base(ATMOSPHERIC_PRESSURE),
                MassDensity::from_base(DENSITY_WATER_NOMINAL),
            )
            .into_base();
        let frequency = frequency.into_base();
        let concentration = concentration.into_base();
        let freq_ratio = frequency / resonance_freq;
        let resonance_factor = 1.0 / (freq_ratio - 1.0).mul_add(freq_ratio - 1.0, 1.0);

        // Total nonlinear scattering
        let nonlinear_scattering = linear_scattering
            * concentration
            * self.harmonic_efficiency.mul_add(pressure_factor, 1.0)
            * resonance_factor;

        Ok(ReciprocalLength::from_base(nonlinear_scattering))
    }
}

/// Harmonic imaging for CEUS
#[derive(Debug)]
pub struct HarmonicImaging {
    /// Harmonic frequencies to extract
    harmonic_frequencies: Vec<Frequency<f64>>,
    /// Imaging parameters
    pub parameters: HarmonicImagingParameters,
}

#[derive(Debug, Clone)]
pub struct HarmonicImagingParameters {
    /// Transmit frequency (Hz)
    pub transmit_freq: Frequency<f64>,
    /// Receive bandwidth (Hz)
    pub bandwidth: Frequency<f64>,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Frame rate (Hz)
    pub frame_rate: Frequency<f64>,
}

impl HarmonicImaging {
    /// Create new harmonic imaging system
    #[must_use]
    pub fn new(fundamental_freq: Frequency<f64>) -> Self {
        let fundamental_freq = fundamental_freq.into_base();
        let harmonic_frequencies = vec![
            Frequency::from_base(fundamental_freq * 2.0), // Second harmonic
            Frequency::from_base(fundamental_freq * 1.5), // Ultraharmonic
            Frequency::from_base(fundamental_freq * 0.5), // Subharmonic
        ];

        Self {
            harmonic_frequencies,
            parameters: HarmonicImagingParameters {
                transmit_freq: Frequency::from_base(fundamental_freq),
                bandwidth: Frequency::from_base(fundamental_freq * 0.5),
                mechanical_index: 0.1,
                frame_rate: Frequency::from_base(15.0),
            },
        }
    }

    /// Extract harmonic components from scattered signal
    #[must_use]
    pub fn extract_harmonics(&self, signal: &[f64], sample_rate: Frequency<f64>) -> Vec<f64> {
        let sample_rate = sample_rate.into_base();
        self.harmonic_frequencies
            .iter()
            .map(|freq| self.extract_single_frequency(signal, freq.into_base(), sample_rate))
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
            let phase = TWO_PI * frequency * i as f64 / sample_rate;
            real += sample * phase.cos();
            imag += sample * phase.sin();
        }

        real.hypot(imag) / n as f64
    }
}
